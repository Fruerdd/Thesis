import os
import math
import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from torch.cuda.amp import autocast, GradScaler

from tqdm.auto import tqdm
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

# -------------------------
# PERF / ENV
# -------------------------
CPU_THREADS = 8
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)

torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -------------------------
# PATHS / DATA
# -------------------------
DATA_DIR = Path("data")
FILE_HEADLINES = DATA_DIR / "allsides_balanced_news_headlines-texts.csv"
FILE_ARTICLES = DATA_DIR / "newsmediabias-full.csv"

OUT_DIR = Path("./bias_system_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# MODEL / TRAIN PARAMS
# -------------------------
BASE_MODEL = "bert-base-uncased"

MAX_LENGTH = 256
MAX_CHUNKS = 2
BATCH_SIZE = 16 if DEVICE == "cuda" else 8

EPOCHS_TEACHER = 2
EPOCHS_STUDENT = 3

LR = 2e-5
WARMUP_RATIO = 0.06

# Loss weights (student)
W_LEAN_HARD = 1.0
W_LEAN_SOFT = 0.7      # KL on pseudo-labels
W_INTENSITY = 1.0
W_DOMAIN = 0.05        # <= option C: reduce domain pressure

DOMAIN_HEADLINES = 0
DOMAIN_ARTICLES = 1

LEAN_CANON = ["Right", "Right-center", "Center", "Left-center", "Left"]
INT_CANON  = ["Highly Biased", "Neutral", "Slightly Biased"]

PSEUDO_CACHE = OUT_DIR / "articles_pseudo_lean.parquet"

# -------------------------
# LABEL NORMALIZATION
# -------------------------
def norm_lean(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in {"right", "conservative"}:
        return "Right"
    if s in {"lean right", "right-center", "right center", "center-right", "centerright"}:
        return "Right-center"
    if s in {"center", "neutral", "centrist"}:
        return "Center"
    if s in {"lean left", "left-center", "left center", "center-left", "centerleft"}:
        return "Left-center"
    if s in {"left", "liberal"}:
        return "Left"
    return None

def norm_intensity(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in {"highly biased", "high"}:
        return "Highly Biased"
    if s in {"neutral", "center", "unbiased"}:
        return "Neutral"
    if s in {"slightly biased", "slight"}:
        return "Slightly Biased"
    return None

# -------------------------
# LIGHT FEATURES (your existing)
# -------------------------
HEDGES = {"may","might","could","allegedly","reportedly","apparently","suggests","claimed","sources","source"}
INTENSIFIERS = {"very","extremely","clearly","obviously","undeniably","shocking","huge","massive","disaster","outrage"}
NEGATIONS = {"not","never","no","none","nothing","nowhere","neither","nor"}

def extract_features(text: str) -> np.ndarray:
    t = text or ""
    low = t.lower()
    words = [w for w in low.replace("\n"," ").split(" ") if w]
    n_words = len(words)
    n_chars = len(t)

    def count_set(ws, sset):
        return sum(1 for w in ws if w.strip(".,!?;:()[]{}\"'") in sset)

    n_excl = t.count("!")
    n_q = t.count("?")
    n_quotes = t.count("\"") + t.count("“") + t.count("”") + t.count("'")
    upper = sum(1 for c in t if c.isupper())
    alpha = sum(1 for c in t if c.isalpha())
    upper_ratio = (upper / max(alpha, 1))

    hedges = count_set(words, HEDGES)
    intens = count_set(words, INTENSIFIERS)
    negs = count_set(words, NEGATIONS)

    avg_word_len = (sum(len(w) for w in words) / max(n_words, 1))
    long_words = sum(1 for w in words if len(w) >= 7)
    long_word_ratio = long_words / max(n_words, 1)

    punct = sum(1 for c in t if c in ".,!?;:-")
    punct_ratio = punct / max(n_chars, 1)

    feats = np.array([
        n_words,
        n_chars,
        avg_word_len,
        long_word_ratio,
        n_excl,
        n_q,
        n_quotes,
        upper_ratio,
        punct_ratio,
        hedges / max(n_words, 1),
        intens / max(n_words, 1),
        negs / max(n_words, 1),
    ], dtype=np.float32)

    feats[0] = math.log1p(feats[0])
    feats[1] = math.log1p(feats[1])
    feats[4] = math.log1p(feats[4])
    feats[5] = math.log1p(feats[5])
    feats[6] = math.log1p(feats[6])
    return feats

# -------------------------
# TOKEN → CHUNKS
# -------------------------
def encode_to_chunks(tokenizer, text: str, max_length: int, max_chunks: int):
    text = "" if text is None else str(text)
    chunk_size = max_length - 2
    max_total = chunk_size * max_chunks

    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_total,
        return_attention_mask=False,
    )

    ids = enc["input_ids"]
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    chunks = []
    for i in range(0, len(ids), chunk_size):
        seg = ids[i:i + chunk_size]
        if len(seg) == 0:
            break
        chunk = [cls_id] + seg + [sep_id]
        chunks.append(chunk)
        if len(chunks) >= max_chunks:
            break

    if not chunks:
        chunks = [[cls_id, sep_id]]

    padded, attn, chunk_mask = [], [], []
    for c in chunks:
        if len(c) < max_length:
            c = c + [pad_id] * (max_length - len(c))
        else:
            c = c[:max_length]
        a = [0 if tok == pad_id else 1 for tok in c]
        padded.append(c)
        attn.append(a)
        chunk_mask.append(1)

    while len(padded) < max_chunks:
        padded.append([pad_id] * max_length)
        attn.append([0] * max_length)
        chunk_mask.append(0)

    return padded, attn, chunk_mask

# -------------------------
# BATCH / COLLATOR
# Includes soft lean labels for pseudo-labeled articles
# -------------------------
@dataclass
class Batch:
    input_ids: torch.Tensor          # (B, C, L)
    attention_mask: torch.Tensor     # (B, C, L)
    chunk_mask: torch.Tensor         # (B, C)
    feats: torch.Tensor              # (B, F)

    y_lean: torch.Tensor             # (B,) hard lean or -100
    y_int: torch.Tensor              # (B,) hard intensity or -100
    domain: torch.Tensor             # (B,)

    lean_soft: torch.Tensor          # (B, n_lean) soft dist, zeros if none
    has_lean_soft: torch.Tensor      # (B,) 1 if soft label exists else 0

class HierCollator:
    def __init__(self, tokenizer, max_length: int, max_chunks: int, n_lean: int):
        self.tok = tokenizer
        self.max_length = max_length
        self.max_chunks = max_chunks
        self.n_lean = n_lean

    def __call__(self, examples: List[Dict[str, Any]]) -> Batch:
        B = len(examples)
        input_ids = torch.zeros((B, self.max_chunks, self.max_length), dtype=torch.long)
        attn = torch.zeros((B, self.max_chunks, self.max_length), dtype=torch.long)
        cmask = torch.zeros((B, self.max_chunks), dtype=torch.long)
        feats = torch.zeros((B, len(examples[0]["feats"])), dtype=torch.float32)

        y_lean = torch.full((B,), -100, dtype=torch.long)
        y_int  = torch.full((B,), -100, dtype=torch.long)
        dom    = torch.zeros((B,), dtype=torch.long)

        lean_soft = torch.zeros((B, self.n_lean), dtype=torch.float32)
        has_soft = torch.zeros((B,), dtype=torch.long)

        for i, ex in enumerate(examples):
            input_ids[i] = torch.tensor(ex["input_ids_chunks"], dtype=torch.long)
            attn[i] = torch.tensor(ex["attention_mask_chunks"], dtype=torch.long)
            cmask[i] = torch.tensor(ex["chunk_mask"], dtype=torch.long)
            feats[i] = torch.tensor(ex["feats"], dtype=torch.float32)

            y_lean[i] = int(ex["y_lean"])
            y_int[i]  = int(ex["y_int"])
            dom[i]    = int(ex["domain"])

            if ex.get("lean_soft") is not None:
                ls = np.array(ex["lean_soft"], dtype=np.float32)
                if ls.shape[0] == self.n_lean and float(ls.sum()) > 0:
                    lean_soft[i] = torch.tensor(ls, dtype=torch.float32)
                    has_soft[i] = 1

        return Batch(
            input_ids=input_ids,
            attention_mask=attn,
            chunk_mask=cmask,
            feats=feats,
            y_lean=y_lean,
            y_int=y_int,
            domain=dom,
            lean_soft=lean_soft,
            has_lean_soft=has_soft,
        )

# -------------------------
# GRL (Domain Adversarial)
# -------------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd: float):
    return GradReverse.apply(x, lambd)

# -------------------------
# MODEL
# Important change for option C:
# Domain classifier uses DOC embedding (encoder output) not fused features
# -------------------------
class HierMultiTaskBiasModel(nn.Module):
    def __init__(self, encoder_name: str, feat_dim: int, n_lean: int, n_int: int, n_domain: int = 2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size

        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h, h),
        )

        self.chunk_attn = nn.Sequential(
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )

        self.gate_lean = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.ReLU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )
        self.gate_int = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.ReLU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(h)

        self.head_lean = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h, n_lean),
        )
        self.head_int = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h, n_int),
        )

        self.domain_disc = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h, n_domain),
        )

    def forward(self, batch: Batch, grl_lambda: float = 1.0):
        B, C, L = batch.input_ids.shape
        x_ids = batch.input_ids.view(B * C, L).to(DEVICE)
        x_att = batch.attention_mask.view(B * C, L).to(DEVICE)

        out = self.encoder(input_ids=x_ids, attention_mask=x_att)
        cls = out.last_hidden_state[:, 0, :]          # (B*C, H)
        H = cls.shape[-1]
        cls = cls.view(B, C, H)                       # (B, C, H)

        scores = self.chunk_attn(cls).squeeze(-1)     # (B, C)
        mask = (batch.chunk_mask.to(DEVICE) > 0)
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=-1)
        doc = torch.sum(cls * w.unsqueeze(-1), dim=1) # (B, H)

        f = self.feat_proj(batch.feats.to(DEVICE))    # (B, H)

        # Task-specific fusion
        gL = self.gate_lean(torch.cat([doc, f], dim=-1))
        fused_lean = self.norm(gL * doc + (1 - gL) * f)

        gI = self.gate_int(torch.cat([doc, f], dim=-1))
        fused_int = self.norm(gI * doc + (1 - gI) * f)

        logits_lean = self.head_lean(fused_lean)
        logits_int  = self.head_int(fused_int)

        # OPTION C: Domain classifier sees encoder doc embedding, not fused
        dom_inp = grad_reverse(doc, grl_lambda)
        logits_dom = self.domain_disc(dom_inp)

        return {
            "logits_lean": logits_lean,
            "logits_int": logits_int,
            "logits_dom": logits_dom,
            "chunk_attn": w.detach(),
        }

# -------------------------
# HELPERS
# -------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def detect_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found. Available: {list(df.columns)}")

def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    targets = targets.to(logits.device)
    mask = targets.ne(ignore_index)
    if mask.sum().item() == 0:
        return torch.zeros((), device=logits.device)
    return F.cross_entropy(logits[mask], targets[mask])

def soft_kld_loss(logits: torch.Tensor, soft_targets: torch.Tensor, has_soft: torch.Tensor) -> torch.Tensor:
    """
    KLDiv between predicted distribution and teacher distribution, only for rows with has_soft==1
    """
    has_soft = has_soft.to(logits.device)
    mask = has_soft.eq(1)
    if mask.sum().item() == 0:
        return torch.zeros((), device=logits.device)

    logp = F.log_softmax(logits[mask], dim=-1)
    tgt = soft_targets[mask].to(logits.device)
    tgt = tgt / (tgt.sum(dim=-1, keepdim=True).clamp_min(1e-8))
    return F.kl_div(logp, tgt, reduction="batchmean")

def evaluate(model: HierMultiTaskBiasModel, dl: DataLoader) -> Dict[str, float]:
    model.eval()
    all_lean_p, all_lean_y = [], []
    all_int_p, all_int_y = [], []
    dom_p, dom_y = [], []
    losses = []

    with torch.no_grad():
        for batch in dl:
            out = model(batch, grl_lambda=0.0)

            l_lean = masked_ce_loss(out["logits_lean"], batch.y_lean, ignore_index=-100)
            l_int = masked_ce_loss(out["logits_int"], batch.y_int, ignore_index=-100)
            l_dom = F.cross_entropy(out["logits_dom"], batch.domain.to(DEVICE))

            loss = (l_lean + l_int + W_DOMAIN*l_dom)
            losses.append(loss.item())

            yL = batch.y_lean.numpy()
            pL = out["logits_lean"].argmax(dim=-1).detach().cpu().numpy()
            m = yL != -100
            if m.any():
                all_lean_y.extend(yL[m].tolist())
                all_lean_p.extend(pL[m].tolist())

            yI = batch.y_int.numpy()
            pI = out["logits_int"].argmax(dim=-1).detach().cpu().numpy()
            m2 = yI != -100
            if m2.any():
                all_int_y.extend(yI[m2].tolist())
                all_int_p.extend(pI[m2].tolist())

            dom_y.extend(batch.domain.numpy().tolist())
            dom_p.extend(out["logits_dom"].argmax(dim=-1).detach().cpu().numpy().tolist())

    metrics = {"eval_loss": float(np.mean(losses))}
    if all_lean_y:
        metrics["lean_acc"] = accuracy_score(all_lean_y, all_lean_p)
        metrics["lean_f1_macro"] = f1_score(all_lean_y, all_lean_p, average="macro")
    if all_int_y:
        metrics["int_acc"] = accuracy_score(all_int_y, all_int_p)
        metrics["int_f1_macro"] = f1_score(all_int_y, all_int_p, average="macro")
    metrics["domain_acc"] = accuracy_score(dom_y, dom_p)
    return metrics

# -------------------------
# DATA LOADING / PREP
# -------------------------
def load_and_prepare_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_h = pd.read_csv(FILE_HEADLINES, low_memory=False)
    text_col_h = detect_col(df_h, ["text", "heading", "title"])
    label_col_h = detect_col(df_h, ["bias_rating", "bias", "label", "class", "stance"])

    df_h["lean"] = df_h[label_col_h].map(norm_lean)
    df_h = df_h[df_h["lean"].notna() & df_h[text_col_h].notna()].copy()
    df_h = df_h.reset_index(drop=True)
    df_h = df_h.rename(columns={text_col_h: "text"})

    lean_to_id = {k:i for i,k in enumerate(LEAN_CANON)}
    df_h["y_lean"] = df_h["lean"].map(lean_to_id).astype(int)
    df_h["y_int"] = -100
    df_h["domain"] = DOMAIN_HEADLINES

    df_a = pd.read_csv(FILE_ARTICLES, low_memory=False)
    text_col_a = detect_col(df_a, ["text"])
    label_col_a = detect_col(df_a, ["label", "bias", "bias_rating", "overall_bias"])

    df_a["intensity"] = df_a[label_col_a].map(norm_intensity)
    df_a = df_a[df_a["intensity"].notna() & df_a[text_col_a].notna()].copy()
    df_a = df_a.reset_index(drop=True)
    df_a = df_a.rename(columns={text_col_a: "text"})

    int_to_id = {k:i for i,k in enumerate(INT_CANON)}
    df_a["y_int"] = df_a["intensity"].map(int_to_id).astype(int)
    df_a["y_lean"] = -100
    df_a["domain"] = DOMAIN_ARTICLES

    return df_h[["text", "y_lean", "y_int", "domain"]], df_a[["text", "y_lean", "y_int", "domain"]]

def map_text_to_model_inputs(tokenizer, df: pd.DataFrame, n_lean: int) -> Dataset:
    ds = Dataset.from_pandas(df.reset_index(drop=True))

    def map_fn(batch):
        texts = batch["text"]
        feats_list, ids_list, att_list, cm_list = [], [], [], []
        for t in texts:
            t = "" if t is None else str(t)
            feats_list.append(extract_features(t).tolist())
            ids, att, cm = encode_to_chunks(tokenizer, t, MAX_LENGTH, MAX_CHUNKS)
            ids_list.append(ids)
            att_list.append(att)
            cm_list.append(cm)

        out = {
            "input_ids_chunks": ids_list,
            "attention_mask_chunks": att_list,
            "chunk_mask": cm_list,
            "feats": feats_list,
        }
        # lean_soft and has_lean_soft columns might exist; keep them if present
        if "lean_soft" in batch:
            out["lean_soft"] = batch["lean_soft"]
        else:
            out["lean_soft"] = [None] * len(texts)

        return out

    ds = ds.map(map_fn, batched=True, batch_size=64, load_from_cache_file=False)
    return ds

# -------------------------
# TEACHER: HEADLINE LEAN ONLY
# -------------------------
def train_teacher_headlines(seed: int, encoder_path: str) -> str:
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(encoder_path)

    df_h, _ = load_and_prepare_raw()

    # split only headlines
    train_df, val_df = train_test_split(df_h, test_size=0.08, random_state=42, stratify=df_h["y_lean"])

    train_ds = map_text_to_model_inputs(tok, train_df, n_lean=len(LEAN_CANON))
    val_ds   = map_text_to_model_inputs(tok, val_df, n_lean=len(LEAN_CANON))

    collator = HierCollator(tok, MAX_LENGTH, MAX_CHUNKS, n_lean=len(LEAN_CANON))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=0)

    feat_dim = len(train_ds[0]["feats"])
    model = HierMultiTaskBiasModel(
        encoder_name=encoder_path,
        feat_dim=feat_dim,
        n_lean=len(LEAN_CANON),
        n_int=len(INT_CANON),
        n_domain=2,
    ).to(DEVICE)

    # We only train leaning head for teacher
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = EPOCHS_TEACHER * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * WARMUP_RATIO), total_steps)

    print("[Teacher] training lean on headlines only...")
    step = 0
    scaler = GradScaler()
    for epoch in range(1, EPOCHS_TEACHER + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"teacher epoch {epoch}/{EPOCHS_TEACHER}", dynamic_ncols=True)
        for batch in pbar:
            step += 1
            out = model(batch, grl_lambda=0.0)  # no DANN in teacher
            l_lean = masked_ce_loss(out["logits_lean"], batch.y_lean, ignore_index=-100)

            l_lean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            pbar.set_postfix({"lean_loss": f"{l_lean.item():.3f}"})

        metrics = evaluate(model, val_dl)
        print(f"[Teacher eval][epoch {epoch}] {metrics}")

    save_dir = OUT_DIR / "teacher_headlines_lean"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")
    tok.save_pretrained(save_dir)
    (save_dir / "meta.json").write_text(json.dumps({"seed": seed, "encoder_path": encoder_path}, indent=2), encoding="utf-8")
    print("Saved teacher to:", str(save_dir))
    return str(save_dir)

def load_model(model_dir: str, encoder_path: str) -> Tuple[AutoTokenizer, HierMultiTaskBiasModel, int]:
    tok = AutoTokenizer.from_pretrained(model_dir)
    feat_dim = 12
    model = HierMultiTaskBiasModel(encoder_name=encoder_path, feat_dim=feat_dim, n_lean=5, n_int=3, n_domain=2).to(DEVICE)
    sd = torch.load(Path(model_dir) / "model.pt", map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return tok, model, feat_dim

@torch.no_grad()
def teacher_predict_proba_lean_batched(
    texts: List[str],
    teacher_dir: str,
    encoder_path: str,
    batch_size: int = 64,
):
    tok, model, _ = load_model(teacher_dir, encoder_path)
    model.eval()

    rows = []
    for t in texts:
        ids, att, cm = encode_to_chunks(tok, t, MAX_LENGTH, MAX_CHUNKS)
        rows.append({
            "input_ids_chunks": ids,
            "attention_mask_chunks": att,
            "chunk_mask": cm,
            "feats": extract_features(t).tolist(),
            "y_lean": -100,
            "y_int": -100,
            "domain": 0,
            "lean_soft": None,
        })

    ds = Dataset.from_list(rows)

    collator = HierCollator(
        tokenizer=tok,
        max_length=MAX_LENGTH,
        max_chunks=MAX_CHUNKS,
        n_lean=len(LEAN_CANON),
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=(DEVICE == "cuda"),
    )

    all_probs = []

    for batch in tqdm(dl, desc="teacher batched inference", dynamic_ncols=True):
        out = model(batch, grl_lambda=0.0)
        probs = torch.softmax(out["logits_lean"], dim=-1)
        all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0).numpy()

# -------------------------
# PSEUDO-LABEL ARTICLES (lean soft labels)
# -------------------------
def pseudo_label_articles(teacher_dir: str, encoder_path: str):
    print("[Pseudo-label] loading raw data...")
    _, df_a = load_and_prepare_raw()

    if PSEUDO_CACHE.exists():
        print("[Pseudo-label] cache exists:", PSEUDO_CACHE)
        return

    texts = df_a["text"].astype(str).tolist()
    print(f"[Pseudo-label] total articles: {len(texts):,}")

    probs = teacher_predict_proba_lean_batched(
        texts=texts,
        teacher_dir=teacher_dir,
        encoder_path=encoder_path,
        batch_size=64 if DEVICE == "cuda" else 16,
    )

    df_out = pd.DataFrame({
        "text": texts,
        "lean_soft": [p.tolist() for p in probs],
    })

    df_out.to_parquet(PSEUDO_CACHE, index=False)
    print("[Pseudo-label] saved to:", PSEUDO_CACHE)

# -------------------------
# STUDENT DATASET: headlines (hard lean) + articles (hard intensity + soft lean)
# -------------------------
def build_student_dataset(tokenizer) -> Tuple[Dataset, Dataset]:
    df_h, df_a = load_and_prepare_raw()

    if not PSEUDO_CACHE.exists():
        raise RuntimeError(f"Pseudo-label cache not found: {PSEUDO_CACHE}. Run --pseudo_label_articles first.")

    pseudo_df = pd.read_parquet(PSEUDO_CACHE)
    # Merge by text (simple), assumes texts match. If not, you can add an ID column in your raw CSVs.
    df_a = df_a.merge(pseudo_df, on="text", how="left")

    # Ensure lean_soft exists
    df_h["lean_soft"] = None
    # For articles, lean_soft should be list[5]
    df_a["lean_soft"] = df_a["lean_soft"].apply(lambda x: x if isinstance(x, list) else None)

    df = pd.concat([df_h, df_a], axis=0, ignore_index=True)

    # Split, stratify by domain to keep both domains
    train_df, val_df = train_test_split(df, test_size=0.05, random_state=42, stratify=df["domain"])

    train_ds = map_text_to_model_inputs(tokenizer, train_df, n_lean=len(LEAN_CANON))
    val_ds   = map_text_to_model_inputs(tokenizer, val_df, n_lean=len(LEAN_CANON))

    # Save labels/meta
    meta = {
        "lean_labels": LEAN_CANON,
        "intensity_labels": INT_CANON,
        "max_length": MAX_LENGTH,
        "max_chunks": MAX_CHUNKS,
    }
    (OUT_DIR / "labels.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return train_ds, val_ds

# -------------------------
# STUDENT TRAIN
# -------------------------
def train_student(seed: int, encoder_path: str, run_name: str) -> str:
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(encoder_path)

    print("[Student] building datasets...")
    train_ds, val_ds = build_student_dataset(tok)

    collator = HierCollator(tok, MAX_LENGTH, MAX_CHUNKS, n_lean=len(LEAN_CANON))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=0,
                          pin_memory=(DEVICE == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=0,
                        pin_memory=(DEVICE == "cuda"))

    feat_dim = len(train_ds[0]["feats"])
    model = HierMultiTaskBiasModel(
        encoder_name=encoder_path,
        feat_dim=feat_dim,
        n_lean=len(LEAN_CANON),
        n_int=len(INT_CANON),
        n_domain=2,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = EPOCHS_STUDENT * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * WARMUP_RATIO), total_steps)

    print("[Student] starting multitask training with hard+soft leaning...")
    step = 0
    for epoch in range(1, EPOCHS_STUDENT + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"student epoch {epoch}/{EPOCHS_STUDENT}", dynamic_ncols=True)
        for batch in pbar:
            step += 1
            p = step / max(total_steps, 1)
            grl_lambda = float(2.0 / (1.0 + math.exp(-10 * p)) - 1.0)

            out = model(batch, grl_lambda=grl_lambda)

            # Lean hard (headlines)
            l_lean_hard = masked_ce_loss(out["logits_lean"], batch.y_lean, ignore_index=-100)
            # Lean soft (articles pseudo-labels)
            l_lean_soft = soft_kld_loss(out["logits_lean"], batch.lean_soft, batch.has_lean_soft)
            # Intensity hard (articles)
            l_int = masked_ce_loss(out["logits_int"], batch.y_int, ignore_index=-100)
            # Domain (small)
            l_dom = F.cross_entropy(out["logits_dom"], batch.domain.to(DEVICE))

            loss = (
                (W_LEAN_HARD * l_lean_hard) +
                (W_LEAN_SOFT * l_lean_soft) +
                (W_INTENSITY * l_int) +
                (W_DOMAIN * l_dom)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "lean_h": f"{l_lean_hard.item():.2f}",
                "lean_s": f"{l_lean_soft.item():.2f}",
                "int": f"{l_int.item():.2f}",
                "dom": f"{l_dom.item():.2f}",
                "grl": f"{grl_lambda:.2f}",
            })

        metrics = evaluate(model, val_dl)
        print(f"[Student eval][epoch {epoch}] {metrics}")

    save_dir = OUT_DIR / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")
    tok.save_pretrained(save_dir)
    (save_dir / "meta.json").write_text(json.dumps({"seed": seed, "encoder_path": encoder_path}, indent=2), encoding="utf-8")
    print("Saved student to:", str(save_dir))
    return str(save_dir)

# -------------------------
# PREDICT (uses last_models.json if present)
# -------------------------
@torch.no_grad()
def predict(text: str, model_dir: str, encoder_path: str, threshold_non_center: float = 0.5):
    tok, model, _ = load_model(model_dir, encoder_path)
    feats = extract_features(text).tolist()
    ids, att, cm = encode_to_chunks(tok, text, MAX_LENGTH, MAX_CHUNKS)

    batch = Batch(
        input_ids=torch.tensor([ids], dtype=torch.long),
        attention_mask=torch.tensor([att], dtype=torch.long),
        chunk_mask=torch.tensor([cm], dtype=torch.long),
        feats=torch.tensor([feats], dtype=torch.float32),
        y_lean=torch.tensor([-100], dtype=torch.long),
        y_int=torch.tensor([-100], dtype=torch.long),
        domain=torch.tensor([0], dtype=torch.long),
        lean_soft=torch.zeros((1, len(LEAN_CANON)), dtype=torch.float32),
        has_lean_soft=torch.zeros((1,), dtype=torch.long),
    )

    out = model(batch, grl_lambda=0.0)
    probs_lean = torch.softmax(out["logits_lean"], dim=-1).squeeze(0).cpu().numpy()
    probs_int  = torch.softmax(out["logits_int"], dim=-1).squeeze(0).cpu().numpy()
    chunk_attn = out["chunk_attn"].squeeze(0).cpu().numpy()

    pred_lean = LEAN_CANON[int(np.argmax(probs_lean))]
    pred_int  = INT_CANON[int(np.argmax(probs_int))]

    p_center = float(probs_lean[LEAN_CANON.index("Center")])
    biased_score = 1.0 - p_center
    biased = biased_score >= threshold_non_center

    return {
        "political_bias": pred_lean,
        "bias_intensity": pred_int,
        "biased": biased,
        "biased_score": float(biased_score),
        "probs_lean": {LEAN_CANON[i]: float(probs_lean[i]) for i in range(5)},
        "probs_int":  {INT_CANON[i]: float(probs_int[i]) for i in range(3)},
        "chunk_attention": chunk_attn.tolist(),
    }

# -------------------------
# MAIN
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=str, default=BASE_MODEL)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--train_teacher", action="store_true")
    ap.add_argument("--pseudo_label_articles", action="store_true")
    ap.add_argument("--train_student", action="store_true")

    ap.add_argument("--predict", type=str, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    print("Device:", DEVICE)
    encoder_path = args.encoder

    teacher_dir = str(OUT_DIR / "teacher_headlines_lean")

    if args.train_teacher:
        teacher_dir = train_teacher_headlines(seed=args.seed, encoder_path=encoder_path)

    if args.pseudo_label_articles:
        if not Path(teacher_dir).exists():
            raise RuntimeError("Teacher model not found. Run with --train_teacher first.")
        pseudo_label_articles(teacher_dir=teacher_dir, encoder_path=encoder_path)

    if args.train_student:
        run_name = f"student_mt_softlean_seed{args.seed}"
        student_dir = train_student(seed=args.seed, encoder_path=encoder_path, run_name=run_name)
        (OUT_DIR / "last_student.json").write_text(json.dumps({"student_dir": student_dir}, indent=2), encoding="utf-8")

    if args.predict is not None:
        last = OUT_DIR / "last_student.json"
        if not last.exists():
            raise RuntimeError("No trained student found. Run --train_student first.")
        student_dir = json.loads(last.read_text(encoding="utf-8"))["student_dir"]
        res = predict(args.predict, model_dir=student_dir, encoder_path=encoder_path, threshold_non_center=args.threshold)
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()