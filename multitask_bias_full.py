import os
import math
import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


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
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

try:
    from captum.attr import LayerIntegratedGradients
    _HAS_CAPTUM = True
except Exception:
    _HAS_CAPTUM = False

CPU_THREADS = 8
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)

torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DATA_DIR = Path("data")
FILE_HEADLINES = DATA_DIR / "allsides_balanced_news_headlines-texts.csv"
FILE_ARTICLES = DATA_DIR / "newsmediabias-full.csv"

BASE_MODEL = "bert-base-uncased"
OUT_DIR = Path("./bias_system_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MAX_SAMPLES_HEADLINES = 100_000
MAX_SAMPLES_ARTICLES  = 100_000
VAL_SIZE = 0.05

MAX_LENGTH = 256          # tokens per chunk
MAX_CHUNKS = 2            # chunks per document (hierarchical)
BATCH_SIZE = 16 if DEVICE == "cuda" else 8
EPOCHS = 3

MAX_TOTAL_TOKENS = (MAX_LENGTH - 2) * MAX_CHUNKS  # сколько токенов мы максимум используем


LR = 2e-5
WARMUP_RATIO = 0.06

W_LEAN = 1.0
W_INTENSITY = 1.0
W_DOMAIN = 0.2

DOMAIN_HEADLINES = 0
DOMAIN_ARTICLES  = 1


LEAN_CANON = ["Right", "Right-center", "Center", "Left-center", "Left"]
INT_CANON  = ["Highly Biased", "Neutral", "Slightly Biased"]

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



@dataclass
class Batch:
    input_ids: torch.Tensor          # (B, C, L)
    attention_mask: torch.Tensor     # (B, C, L)
    chunk_mask: torch.Tensor         # (B, C)
    feats: torch.Tensor              # (B, F)
    y_lean: torch.Tensor             # (B,)
    y_int: torch.Tensor              # (B,)
    domain: torch.Tensor             # (B,)


class HierCollator:
    def __init__(self, tokenizer, max_length: int, max_chunks: int):
        self.tok = tokenizer
        self.max_length = max_length
        self.max_chunks = max_chunks

    def __call__(self, examples: List[Dict[str, Any]]) -> Batch:
        B = len(examples)
        input_ids = torch.zeros((B, self.max_chunks, self.max_length), dtype=torch.long)
        attn = torch.zeros((B, self.max_chunks, self.max_length), dtype=torch.long)
        cmask = torch.zeros((B, self.max_chunks), dtype=torch.long)
        feats = torch.zeros((B, len(examples[0]["feats"])), dtype=torch.float32)

        y_lean = torch.full((B,), -100, dtype=torch.long)
        y_int  = torch.full((B,), -100, dtype=torch.long)
        dom    = torch.zeros((B,), dtype=torch.long)

        for i, ex in enumerate(examples):
            input_ids[i] = torch.tensor(ex["input_ids_chunks"], dtype=torch.long)
            attn[i] = torch.tensor(ex["attention_mask_chunks"], dtype=torch.long)
            cmask[i] = torch.tensor(ex["chunk_mask"], dtype=torch.long)
            feats[i] = torch.tensor(ex["feats"], dtype=torch.float32)
            y_lean[i] = int(ex["y_lean"])
            y_int[i]  = int(ex["y_int"])
            dom[i]    = int(ex["domain"])

        return Batch(
            input_ids=input_ids,
            attention_mask=attn,
            chunk_mask=cmask,
            feats=feats,
            y_lean=y_lean,
            y_int=y_int,
            domain=dom,
        )


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

        self.gate = nn.Sequential(
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
        cls = out.last_hidden_state[:, 0, :]
        H = cls.shape[-1]
        cls = cls.view(B, C, H)

        scores = self.chunk_attn(cls).squeeze(-1)
        mask = (batch.chunk_mask.to(DEVICE) > 0)
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=-1)
        doc = torch.sum(cls * w.unsqueeze(-1), dim=1)

        f = self.feat_proj(batch.feats.to(DEVICE))

        g = self.gate(torch.cat([doc, f], dim=-1))
        fused = g * doc + (1 - g) * f
        fused = self.norm(fused)

        logits_lean = self.head_lean(fused)
        logits_int  = self.head_int(fused)

        dom_inp = grad_reverse(fused, grl_lambda)
        logits_dom = self.domain_disc(dom_inp)

        return {
            "logits_lean": logits_lean,
            "logits_int": logits_int,
            "logits_dom": logits_dom,
            "chunk_attn": w.detach(),
        }


def detect_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found. Available: {list(df.columns)}")

def build_dataset(tokenizer):
    df_h = pd.read_csv(FILE_HEADLINES, low_memory=False)
    text_col_h = detect_col(df_h, ["text", "heading", "title"])
    label_col_h = detect_col(df_h, ["bias_rating", "bias", "label", "class", "stance"])

    df_h["lean"] = df_h[label_col_h].map(norm_lean)
    df_h = df_h[df_h["lean"].notna() & df_h[text_col_h].notna()].copy()
    if len(df_h) > MAX_SAMPLES_HEADLINES:
        df_h, _ = train_test_split(df_h, train_size=MAX_SAMPLES_HEADLINES, random_state=42, stratify=df_h["lean"])
    df_h = df_h.reset_index(drop=True)

    lean_to_id = {k:i for i,k in enumerate(LEAN_CANON)}
    df_h["y_lean"] = df_h["lean"].map(lean_to_id).astype(int)
    df_h["y_int"] = -100  # missing
    df_h["domain"] = DOMAIN_HEADLINES

    df_a = pd.read_csv(FILE_ARTICLES, low_memory=False)
    text_col_a = detect_col(df_a, ["text"])
    label_col_a = detect_col(df_a, ["label", "bias", "bias_rating", "overall_bias"])

    df_a["intensity"] = df_a[label_col_a].map(norm_intensity)
    df_a = df_a[df_a["intensity"].notna() & df_a[text_col_a].notna()].copy()
    if len(df_a) > MAX_SAMPLES_ARTICLES:
        df_a, _ = train_test_split(df_a, train_size=MAX_SAMPLES_ARTICLES, random_state=42, stratify=df_a["intensity"])
    df_a = df_a.reset_index(drop=True)

    int_to_id = {k:i for i,k in enumerate(INT_CANON)}
    df_a["y_int"] = df_a["intensity"].map(int_to_id).astype(int)
    df_a["y_lean"] = -100  # missing
    df_a["domain"] = DOMAIN_ARTICLES

    df_h = df_h[[text_col_h, "y_lean", "y_int", "domain"]].rename(columns={text_col_h: "text"})
    df_a = df_a[[text_col_a, "y_lean", "y_int", "domain"]].rename(columns={text_col_a: "text"})
    df = pd.concat([df_h, df_a], axis=0, ignore_index=True)

    train_df, val_df = train_test_split(df, test_size=VAL_SIZE, random_state=42, stratify=df["domain"])

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    def map_fn(ex):
        text = ex["text"]
        feats = extract_features(text).tolist()
        ids, att, cm = encode_to_chunks(tokenizer, text, MAX_LENGTH, MAX_CHUNKS)
        return {
            "input_ids_chunks": ids,
            "attention_mask_chunks": att,
            "chunk_mask": cm,
            "feats": feats,
        }

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

        return {
            "input_ids_chunks": ids_list,
            "attention_mask_chunks": att_list,
            "chunk_mask": cm_list,
            "feats": feats_list,
        }

    train_ds = train_ds.map(
        map_fn,
        batched=True,
        batch_size=64,
        load_from_cache_file=False,
    )
    val_ds = val_ds.map(
        map_fn,
        batched=True,
        batch_size=64,
        load_from_cache_file=False,
    )

    meta = {
        "lean_labels": LEAN_CANON,
        "intensity_labels": INT_CANON,
        "max_length": MAX_LENGTH,
        "max_chunks": MAX_CHUNKS,
    }
    (OUT_DIR / "labels.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return train_ds, val_ds


def run_dapt(tokenizer, steps: int = 5000, mlm_prob: float = 0.15) -> str:

    out = OUT_DIR / "bert_dapt"
    out.mkdir(parents=True, exist_ok=True)

    df_h = pd.read_csv(FILE_HEADLINES, low_memory=False)
    df_a = pd.read_csv(FILE_ARTICLES, low_memory=False)

    text_h = "text" if "text" in df_h.columns else ("heading" if "heading" in df_h.columns else "title")
    text_a = "text"

    texts = []
    texts.extend(df_h[text_h].dropna().astype(str).tolist()[:200_000])
    texts.extend(df_a[text_a].dropna().astype(str).tolist()[:200_000])

    random.shuffle(texts)
    unl = Dataset.from_dict({"text": texts})

    def tok_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH, padding=False)

    unl = unl.map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL).to(DEVICE)
    model.train()

    dl = DataLoader(unl, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = steps
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(total_steps*WARMUP_RATIO), num_training_steps=total_steps)

    it = iter(dl)
    for step in range(1, total_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outp = model(**batch)
        loss = outp.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if step % 200 == 0:
            print(f"[DAPT] step {step}/{total_steps} loss={loss.item():.4f}")

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print("Saved DAPT encoder to:", str(out))
    return str(out)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:

    targets = targets.to(logits.device)
    mask = targets.ne(ignore_index)

    if mask.sum().item() == 0:
        return torch.zeros((), device=logits.device)

    return F.cross_entropy(logits[mask], targets[mask])

def evaluate(model: HierMultiTaskBiasModel, dl: DataLoader) -> Dict[str, float]:
    model.eval()
    all_lean_p, all_lean_y = [], []
    all_int_p, all_int_y = [], []
    dom_p, dom_y = [], []
    losses = []


    with torch.no_grad():
        for batch in dl:
            out = model(batch, grl_lambda=1.0)

            l_lean = masked_ce_loss(out["logits_lean"], batch.y_lean, ignore_index=-100)
            l_int = masked_ce_loss(out["logits_int"], batch.y_int, ignore_index=-100)
            l_dom = F.cross_entropy(out["logits_dom"], batch.domain.to(DEVICE))

            loss = (W_LEAN*l_lean) + (W_INTENSITY*l_int) + (W_DOMAIN*l_dom)
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


def train_one(seed: int, encoder_path: str, run_name: str):
    set_seed(seed)

    tok = AutoTokenizer.from_pretrained(encoder_path)

    print("[1/5] Building datasets...")
    t0 = time.time()
    train_ds, val_ds = build_dataset(tok)
    print(f"[1/5] Done. train={len(train_ds):,} val={len(val_ds):,} in {time.time() - t0:.1f}s")

    print("[2/5] Building dataloaders...")
    collator = HierCollator(tok, MAX_LENGTH, MAX_CHUNKS)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=0,
                          pin_memory=(DEVICE == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=0,
                        pin_memory=(DEVICE == "cuda"))
    print(f"[2/5] Done. steps/epoch={len(train_dl):,}")

    feat_dim = len(train_ds[0]["feats"])
    model = HierMultiTaskBiasModel(
        encoder_name=encoder_path,
        feat_dim=feat_dim,
        n_lean=len(LEAN_CANON),
        n_int=len(INT_CANON),
        n_domain=2,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = EPOCHS * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(total_steps*WARMUP_RATIO), num_training_steps=total_steps)

    ce_lean = nn.CrossEntropyLoss(ignore_index=-100)
    ce_int  = nn.CrossEntropyLoss(ignore_index=-100)
    ce_dom  = nn.CrossEntropyLoss()

    model.train()
    step = 0

    print("[3/5] Loading model...")
    t0 = time.time()

    feat_dim = len(train_ds[0]["feats"])
    model = HierMultiTaskBiasModel(
        encoder_name=encoder_path,
        feat_dim=feat_dim,
        n_lean=len(LEAN_CANON),
        n_int=len(INT_CANON),
        n_domain=2,
    ).to(DEVICE)

    print(f"[3/5] Done. in {time.time() - t0:.1f}s")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = EPOCHS * len(train_dl)
    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps
    )

    ce_lean = nn.CrossEntropyLoss(ignore_index=-100)
    ce_int = nn.CrossEntropyLoss(ignore_index=-100)
    ce_dom = nn.CrossEntropyLoss()

    print("[4/5] Starting training...")
    model.train()
    step = 0

    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(train_dl, desc=f"train epoch {epoch}/{EPOCHS}", dynamic_ncols=True)
        for batch in pbar:
            step += 1

            p = step / max(total_steps, 1)
            grl_lambda = float(2.0 / (1.0 + math.exp(-10 * p)) - 1.0)

            out = model(batch, grl_lambda=grl_lambda)
            l_lean = masked_ce_loss(out["logits_lean"], batch.y_lean, ignore_index=-100)
            l_int = masked_ce_loss(out["logits_int"], batch.y_int, ignore_index=-100)
            l_dom = F.cross_entropy(out["logits_dom"], batch.domain.to(DEVICE))

            loss = (W_LEAN * l_lean) + (W_INTENSITY * l_int) + (W_DOMAIN * l_dom)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "lean": f"{l_lean.item():.2f}",
                "int": f"{l_int.item():.2f}",
                "dom": f"{l_dom.item():.2f}",
                "grl": f"{grl_lambda:.2f}",
            })

        metrics = evaluate(model, val_dl)
        print(f"[eval][epoch {epoch}] {metrics}")

    print("[5/5] Training finished.")

    save_dir = OUT_DIR / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")
    tok.save_pretrained(save_dir)
    (save_dir / "meta.json").write_text(json.dumps({"seed": seed, "encoder_path": encoder_path}, indent=2), encoding="utf-8")
    print("Saved model to:", str(save_dir))
    return str(save_dir)

def load_model(model_dir: str, encoder_path: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    feat_dim = 12
    model = HierMultiTaskBiasModel(encoder_name=encoder_path, feat_dim=feat_dim, n_lean=5, n_int=3, n_domain=2).to(DEVICE)
    sd = torch.load(Path(model_dir) / "model.pt", map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return tok, model

@torch.no_grad()
def predict(text: str, model_dirs: List[str], encoder_path: str, threshold_non_center: float = 0.5):
    probs_lean = None
    probs_int  = None
    chunk_attn = None

    for md in model_dirs:
        tok, model = load_model(md, encoder_path)
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
        )

        out = model(batch, grl_lambda=0.0)
        pl = torch.softmax(out["logits_lean"], dim=-1).squeeze(0).cpu().numpy()
        pi = torch.softmax(out["logits_int"], dim=-1).squeeze(0).cpu().numpy()

        probs_lean = pl if probs_lean is None else (probs_lean + pl)
        probs_int  = pi if probs_int is None else (probs_int + pi)
        chunk_attn = out["chunk_attn"].squeeze(0).cpu().numpy()

    probs_lean /= len(model_dirs)
    probs_int  /= len(model_dirs)

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



def explain_ig(text: str, model_dir: str, encoder_path: str, target: str = "lean", target_class: Optional[int] = None):
    if not _HAS_CAPTUM:
        raise RuntimeError("captum not installed. pip install captum")

    from captum.attr import IntegratedGradients

    tok, model = load_model(model_dir, encoder_path)
    model.eval()

    ids, att, cm = encode_to_chunks(tok, text, MAX_LENGTH, MAX_CHUNKS)
    first_ids = torch.tensor([ids[0]], dtype=torch.long).to(DEVICE)  # (1, L)
    first_att = torch.tensor([att[0]], dtype=torch.long).to(DEVICE)  # (1, L)

    emb_layer = model.encoder.embeddings.word_embeddings
    inp_emb = emb_layer(first_ids)  # (1, L, H)

    def forward_emb(embeddings):
        N = embeddings.size(0)

        attn = first_att.expand(N, -1)  # (N, L)

        out = model.encoder(inputs_embeds=embeddings, attention_mask=attn)
        cls = out.last_hidden_state[:, 0, :]  # (N, H)

        # Make feats match batch size N
        feats = torch.zeros((N, 12), device=embeddings.device)
        f = model.feat_proj(feats)
        g = model.gate(torch.cat([cls, f], dim=-1))
        fused = model.norm(g * cls + (1 - g) * f)

        logits = model.head_lean(fused) if target == "lean" else model.head_int(fused)
        return logits

    with torch.no_grad():
        logits0 = forward_emb(inp_emb)
        if target_class is None:
            target_class = int(torch.argmax(logits0, dim=-1).item())

    ig = IntegratedGradients(forward_emb)

    baseline = torch.zeros_like(inp_emb)
    attributions = ig.attribute(
        inputs=inp_emb,
        baselines=baseline,
        target=target_class,
        n_steps=24,
    )

    scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()  # (L,)
    toks = tok.convert_ids_to_tokens(first_ids.squeeze(0).detach().cpu().tolist())

    items = [(t, float(s)) for t, s in zip(toks, scores) if t != tok.pad_token]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return {"target": target, "target_class": int(target_class), "top_tokens": items[:25]}




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dapt", action="store_true", help="Run domain-adaptive MLM pretraining (DAPT) first")
    ap.add_argument("--dapt_steps", type=int, default=3000)
    ap.add_argument("--train", action="store_true", help="Train multitask hierarchical + DANN model")
    ap.add_argument("--ensemble", type=int, default=1, help="Train N seeds for ensemble (1 = single model)")
    ap.add_argument("--predict", type=str, default=None, help="Text to predict")
    ap.add_argument("--explain_ig", action="store_true", help="Return Integrated Gradients explanation (first chunk)")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    print("Device:", DEVICE)

    encoder_path = BASE_MODEL
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    if args.dapt:
        encoder_path = run_dapt(tok, steps=args.dapt_steps)

    model_dirs = []
    if args.train:
        for i in range(args.ensemble):
            seed = 42 + i * 13
            run_name = f"mt_hier_dann_seed{seed}"
            md = train_one(seed=seed, encoder_path=encoder_path, run_name=run_name)
            model_dirs.append(md)
        (OUT_DIR / "last_models.json").write_text(json.dumps(model_dirs, indent=2), encoding="utf-8")

    if args.predict is not None:
        if not model_dirs:
            lm = OUT_DIR / "last_models.json"
            if lm.exists():
                model_dirs = json.loads(lm.read_text(encoding="utf-8"))
            else:
                model_dirs = [str(OUT_DIR / "mt_hier_dann_seed42")]

        res = predict(args.predict, model_dirs=model_dirs, encoder_path=encoder_path, threshold_non_center=args.threshold)
        print(json.dumps(res, indent=2))

        if args.explain_ig:
            if _HAS_CAPTUM:
                exp = explain_ig(args.predict, model_dir=model_dirs[0], encoder_path=encoder_path, target="lean")
                print("\n[IG explanation]\n", json.dumps(exp, indent=2))
            else:
                print("\nInstall captum to use IG: pip install captum")

if __name__ == "__main__":
    main()
