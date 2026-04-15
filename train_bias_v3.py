import os
import re
import math
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


CPU_THREADS = 4
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

USE_AMP = DEVICE in {"cuda", "mps"}
AMP_DTYPE = torch.float16 if DEVICE in {"cuda", "mps"} else None

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    from torch.amp import GradScaler as _GradScaler

    def make_grad_scaler():
        return _GradScaler("cuda", enabled=(DEVICE == "cuda"))
except Exception:
    from torch.cuda.amp import GradScaler as _GradScaler

    def make_grad_scaler():
        return _GradScaler(enabled=(DEVICE == "cuda"))


DATA_DIR = Path("data")
PREP_DIR = Path("data_prepared")

FILE_COMBINED_LEAN = PREP_DIR / "combined_lean_train_without_holdout.csv"
FILE_NEWSMEDIABIAS = DATA_DIR / "newsmediabias-full.csv"
FILE_ALLSIDES_SOURCES = DATA_DIR / "allsides.csv"

OUT_DIR = Path("./bias_system_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "bert-base-uncased"
MAX_LENGTH = 128
MAX_CHUNKS = 3

BATCH_SIZE = 8 if DEVICE == "cuda" else 4
GRAD_ACCUM = 2
INFER_BATCH = 64 if DEVICE == "cuda" else 16
NUM_WORKERS = 0

EPOCHS_TEACHER = 2
EPOCHS_STUDENT = 1

LR = 2e-5
WARMUP_RATIO = 0.06

W_LEAN_HARD = 2.0
W_LEAN_SOFT = 0.08
W_INTENSITY = 1.0
W_DOMAIN = 0.0

DOMAIN_LEAN = 0
DOMAIN_INTENSITY = 1

# Cap newsmediabias (sentence-level) per intensity class so it doesn't drown out
# the article-level intensity signal derived from the lean dataset (74.5 K rows).
# 15 000 per class ≈ 45 K sentences, vs ~74 K articles → roughly 40/60 split.
NEWSMEDIABIAS_MAX_PER_CLASS = 15_000

# Intensity thresholds calibrated on the real holdout set (100 articles per lean class).
# biased_score = 1 - p_center.  Distribution on holdout: median=0.97, p20=0.75.
# Center articles have median biased_score=0.43; all other classes cluster above 0.85.
# These thresholds give roughly 17% Neutral / 33% Slightly Biased / 50% Highly Biased,
# which matches realistic news-article intensity expectations.
INTENSITY_THRESH_NEUTRAL  = 0.50   # bs < 0.50  → Neutral
INTENSITY_THRESH_SLIGHTLY = 0.97   # bs < 0.97  → Slightly Biased, else Highly Biased

PSEUDO_MIN_CONF = 0.55
SOURCE_PRIOR_ALPHA = 0.10
SOURCE_PRIOR_MARGIN = 0.06

BREAK_EVERY_HOURS = 3
BREAK_DURATION_MIN = 15
ENABLE_TRAINING_BREAKS = False

LEAN_CANON = ["Right", "Right-center", "Center", "Left-center", "Left"]
INT_CANON = ["Highly Biased", "Neutral", "Slightly Biased"]

LEAN_TO_ID = {k: i for i, k in enumerate(LEAN_CANON)}
INT_TO_ID = {k: i for i, k in enumerate(INT_CANON)}

PSEUDO_CACHE = OUT_DIR / "articles_pseudo_lean.parquet"
PSEUDO_CACHE_CSV = OUT_DIR / "articles_pseudo_lean.csv.gz"
PSEUDO_PARTS_DIR = OUT_DIR / "pseudo_parts"

BAD_TEXT_VALUES = {
    "",
    "null",
    "none",
    "nan",
    "error fetching article",
    "<null>",
    "n/a",
}

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


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
    if s in {"highly biased", "high", "toxic"}:
        return "Highly Biased"
    if s in {"neutral", "center", "unbiased"}:
        return "Neutral"
    if s in {"slightly biased", "slight"}:
        return "Slightly Biased"
    return None


def normalize_term(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())



HEDGES = {
    "allegedly", "apparently", "arguably", "assume", "assumed", "assumes",
    "could", "doubtful", "estimated", "fairly", "generally", "likely",
    "mainly", "may", "maybe", "might", "mostly", "often", "perhaps",
    "plausible", "plausibly", "possible", "possibly", "postulated",
    "presumably", "probable", "probably", "purported", "purportedly",
    "quite", "rather", "relatively", "reportedly", "rumored", "seem",
    "seemed", "seemingly", "seems", "somewhat", "suggest", "suggested",
    "suggesting", "suggests", "supposedly", "typically", "uncertain",
    "unclear", "usually", "virtually", "appears", "claimed", "claims",
    "sources say", "it seems", "in part", "sort of", "kind of",
}

INTENSIFIERS = {
    "very", "extremely", "clearly", "obviously", "undeniably", "shocking",
    "huge", "massive", "disaster", "outrage", "awfully", "extraordinary",
    "unusual", "much", "rather", "entirely", "greatly", "really",
    "exceedingly", "too", "completely", "terribly", "perfectly", "quite",
    "certainly", "especially", "fairly", "highly", "increasingly",
    "much more", "particularly", "probably", "more", "absolutely",
    "intensely", "supremely", "most", "pretty"
}

NEGATIONS = {
    "no", "not", "none", "never", "neither", "nobody", "nothing", "nowhere",
    "seldom", "scarcely", "hardly", "barely", "is not", "cannot", "may not",
    "could not", "would not", "did not", "do not", "does not", "was not",
    "are not", "were not"
}


def tokenize_words(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def count_terms(text: str, terms: set) -> int:
    t = f" {normalize_term(text)} "
    count = 0
    for term in terms:
        if " " in term:
            count += t.count(f" {term} ")
        else:
            count += len(re.findall(rf"\b{re.escape(term)}\b", t))
    return count


def source_key(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return normalize_term(str(x))


def clean_text_value(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in BAD_TEXT_VALUES:
        return ""
    return s


def is_valid_text(x: Any, min_len: int = 30) -> bool:
    s = clean_text_value(x)
    if not s:
        return False
    if len(s) < min_len:
        return False
    alpha = sum(1 for c in s if c.isalpha())
    return alpha >= 15


def detect_col(df: pd.DataFrame, candidates: List[str]) -> str:
    col_map = {str(col).strip().lower(): col for col in df.columns}
    for c in candidates:
        key = str(c).strip().lower()
        if key in col_map:
            return col_map[key]
    raise ValueError(f"None of {candidates} found. Available: {list(df.columns)}")


def load_source_bias_map() -> Dict[str, np.ndarray]:
    if not FILE_ALLSIDES_SOURCES.exists():
        return {}
    df = pd.read_csv(FILE_ALLSIDES_SOURCES, low_memory=False)
    try:
        name_col = detect_col(df, ["name", "source", "media", "outlet"])
        bias_col = detect_col(df, ["bias", "bias_rating", "rating", "label", "lean"])
    except Exception:
        return {}
    priors = {}
    for _, row in df.iterrows():
        name = source_key(row[name_col])
        lean = norm_lean(row[bias_col])
        if not name or lean is None:
            continue
        vec = np.full(len(LEAN_CANON), 0.025, dtype=np.float32)
        vec[LEAN_TO_ID[lean]] = 0.90
        vec = vec / vec.sum()
        priors[name] = vec
    return priors


SOURCE_BIAS_MAP = load_source_bias_map()


def apply_source_prior_if_ambiguous(
    probs: np.ndarray,
    source_name: Optional[str],
    alpha: float = SOURCE_PRIOR_ALPHA,
    margin: float = SOURCE_PRIOR_MARGIN,
) -> np.ndarray:
    if source_name is None:
        return probs
    src = source_key(source_name)
    if not src or src not in SOURCE_BIAS_MAP:
        return probs
    order = np.argsort(probs)[::-1]
    top1, top2 = probs[order[0]], probs[order[1]]
    if (top1 - top2) > margin:
        return probs
    prior = SOURCE_BIAS_MAP[src]
    mixed = (1.0 - alpha) * probs + alpha * prior
    mixed = mixed / mixed.sum()
    return mixed


def extract_features(text: str) -> np.ndarray:
    t = text or ""
    low = t.lower()
    words = tokenize_words(low)

    n_words = len(words)
    n_chars = len(t)
    n_excl = t.count("!")
    n_q = t.count("?")
    n_quotes = t.count('"') + t.count("“") + t.count("”") + t.count("'")
    upper = sum(1 for c in t if c.isupper())
    alpha = sum(1 for c in t if c.isalpha())
    upper_ratio = upper / max(alpha, 1)
    hedges = count_terms(low, HEDGES)
    intens = count_terms(low, INTENSIFIERS)
    negs = count_terms(low, NEGATIONS)
    avg_word_len = sum(len(w) for w in words) / max(n_words, 1)
    long_words = sum(1 for w in words if len(w) >= 7)
    long_word_ratio = long_words / max(n_words, 1)
    punct = sum(1 for c in t if c in ".,!?;:-")
    punct_ratio = punct / max(n_chars, 1)
    sentence_count = max(1, len(re.findall(r"[.!?]+", t)))
    exclaim_ratio = n_excl / sentence_count
    question_ratio = n_q / sentence_count

    return np.array(
        [
            math.log1p(n_words),
            math.log1p(n_chars),
            avg_word_len,
            long_word_ratio,
            math.log1p(n_excl),
            math.log1p(n_q),
            math.log1p(n_quotes),
            upper_ratio,
            punct_ratio,
            hedges / max(n_words, 1),
            intens / max(n_words, 1),
            negs / max(n_words, 1),
            exclaim_ratio,
            question_ratio,
        ],
        dtype=np.float32,
    )


def encode_ids_to_chunks(ids: List[int], cls_id: int, sep_id: int, pad_id: int, max_length: int, max_chunks: int):
    chunk_size = max_length - 2
    chunks = []
    for i in range(0, len(ids), chunk_size):
        seg = ids[i:i + chunk_size]
        if not seg:
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
    cls_id = int(tokenizer.cls_token_id)
    sep_id = int(tokenizer.sep_token_id)
    pad_id = int(tokenizer.pad_token_id)
    return encode_ids_to_chunks(ids, cls_id, sep_id, pad_id, max_length, max_chunks)


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    chunk_mask: torch.Tensor
    feats: torch.Tensor
    y_lean: torch.Tensor
    y_int: torch.Tensor
    domain: torch.Tensor
    lean_soft: torch.Tensor
    has_lean_soft: torch.Tensor
    source_name: List[str]


class StudentTextDataset(TorchDataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "text": "" if row["text"] is None else str(row["text"]),
            "y_lean": int(row["y_lean"]),
            "y_int": int(row["y_int"]),
            "domain": int(row["domain"]),
            "lean_soft": row["lean_soft"] if "lean_soft" in row else None,
            "source_name": "" if "source_name" not in row or pd.isna(row["source_name"]) else str(row["source_name"]),
        }


class PseudoTextDataset(TorchDataset):
    def __init__(self, df_articles: pd.DataFrame):
        self.df = df_articles.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return {
            "row_id": int(row["row_id"]),
            "text": "" if row["text"] is None else str(row["text"]),
            "source_name": "" if "source_name" not in row or pd.isna(row["source_name"]) else str(row["source_name"]),
        }


class StudentHierTextCollator:
    def __init__(self, tokenizer, max_length: int, max_chunks: int):
        self.tok = tokenizer
        self.max_length = max_length
        self.max_chunks = max_chunks
        self.cls_id = int(tokenizer.cls_token_id)
        self.sep_id = int(tokenizer.sep_token_id)
        self.pad_id = int(tokenizer.pad_token_id)
        self.max_total = (max_length - 2) * max_chunks

    def __call__(self, examples):
        texts = [ex["text"] for ex in examples]
        enc = self.tok(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_total,
            return_attention_mask=False,
        )
        ids_list = enc["input_ids"]

        B = len(texts)
        C = self.max_chunks
        L = self.max_length

        input_ids = torch.zeros((B, C, L), dtype=torch.long)
        attn = torch.zeros((B, C, L), dtype=torch.long)
        cmask = torch.zeros((B, C), dtype=torch.long)
        feats = torch.zeros((B, 14), dtype=torch.float32)
        y_lean = torch.zeros((B,), dtype=torch.long)
        y_int = torch.zeros((B,), dtype=torch.long)
        domain = torch.zeros((B,), dtype=torch.long)
        lean_soft = torch.zeros((B, len(LEAN_CANON)), dtype=torch.float32)
        has_soft = torch.zeros((B,), dtype=torch.long)
        source_names = []

        for i, (ex, ids) in enumerate(zip(examples, ids_list)):
            padded, am, cm = encode_ids_to_chunks(ids, self.cls_id, self.sep_id, self.pad_id, L, C)
            input_ids[i] = torch.tensor(padded, dtype=torch.long)
            attn[i] = torch.tensor(am, dtype=torch.long)
            cmask[i] = torch.tensor(cm, dtype=torch.long)
            feats[i] = torch.tensor(extract_features(ex["text"]), dtype=torch.float32)
            y_lean[i] = int(ex["y_lean"])
            y_int[i] = int(ex["y_int"])
            domain[i] = int(ex["domain"])
            ls = ex.get("lean_soft", None)
            if ls is not None:
                arr = torch.tensor(ls, dtype=torch.float32)
                if arr.numel() == len(LEAN_CANON) and float(arr.sum()) > 0:
                    lean_soft[i] = arr
                    has_soft[i] = 1
            source_names.append(ex.get("source_name", ""))

        return Batch(
            input_ids=input_ids,
            attention_mask=attn,
            chunk_mask=cmask,
            feats=feats,
            y_lean=y_lean,
            y_int=y_int,
            domain=domain,
            lean_soft=lean_soft,
            has_lean_soft=has_soft,
            source_name=source_names,
        )


class HierTextCollator:
    def __init__(self, tokenizer, max_length: int, max_chunks: int):
        self.tok = tokenizer
        self.max_length = max_length
        self.max_chunks = max_chunks
        self.cls_id = int(tokenizer.cls_token_id)
        self.sep_id = int(tokenizer.sep_token_id)
        self.pad_id = int(tokenizer.pad_token_id)
        self.max_total = (max_length - 2) * max_chunks

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [ex["text"] for ex in examples]
        row_ids = [ex["row_id"] for ex in examples]
        source_names = [ex.get("source_name", "") for ex in examples]

        enc = self.tok(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_total,
            return_attention_mask=False,
        )
        ids_list = enc["input_ids"]

        B = len(texts)
        C = self.max_chunks
        L = self.max_length

        input_ids = torch.zeros((B, C, L), dtype=torch.long)
        attn = torch.zeros((B, C, L), dtype=torch.long)
        cmask = torch.zeros((B, C), dtype=torch.long)
        feats = torch.zeros((B, 14), dtype=torch.float32)

        for i, ids in enumerate(ids_list):
            padded, am, cm = encode_ids_to_chunks(ids, self.cls_id, self.sep_id, self.pad_id, L, C)
            input_ids[i] = torch.tensor(padded, dtype=torch.long)
            attn[i] = torch.tensor(am, dtype=torch.long)
            cmask[i] = torch.tensor(cm, dtype=torch.long)
            feats[i] = torch.tensor(extract_features(texts[i]), dtype=torch.float32)

        return {
            "row_id": torch.tensor(row_ids, dtype=torch.long),
            "source_name": source_names,
            "batch": Batch(
                input_ids=input_ids,
                attention_mask=attn,
                chunk_mask=cmask,
                feats=feats,
                y_lean=torch.full((B,), -100, dtype=torch.long),
                y_int=torch.full((B,), -100, dtype=torch.long),
                domain=torch.full((B,), DOMAIN_INTENSITY, dtype=torch.long),
                lean_soft=torch.zeros((B, len(LEAN_CANON)), dtype=torch.float32),
                has_lean_soft=torch.zeros((B,), dtype=torch.long),
                source_name=source_names,
            ),
        }


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
        cls = out.last_hidden_state[:, 0, :]
        H = cls.shape[-1]
        cls = cls.view(B, C, H)

        scores = self.chunk_attn(cls).squeeze(-1)
        mask = batch.chunk_mask.to(DEVICE).bool()
        scores_f = scores.float().masked_fill(~mask, -1e9)
        w = torch.softmax(scores_f, dim=-1).to(cls.dtype)

        doc = torch.sum(cls * w.unsqueeze(-1), dim=1)
        f = self.feat_proj(batch.feats.to(DEVICE))

        gL = self.gate_lean(torch.cat([doc, f], dim=-1))
        fused_lean = self.norm(gL * doc + (1 - gL) * f)

        gI = self.gate_int(torch.cat([doc, f], dim=-1))
        fused_int = self.norm(gI * doc + (1 - gI) * f)

        logits_lean = self.head_lean(fused_lean)
        logits_int = self.head_int(fused_int)

        dom_inp = grad_reverse(doc, grl_lambda)
        logits_dom = self.domain_disc(dom_inp)

        return {
            "logits_lean": logits_lean,
            "logits_int": logits_int,
            "logits_dom": logits_dom,
            "chunk_attn": w.detach(),
        }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    targets = targets.to(logits.device)
    mask = targets.ne(ignore_index)
    if mask.sum().item() == 0:
        return torch.zeros((), device=logits.device)
    return F.cross_entropy(logits[mask], targets[mask])


def masked_ce_loss_weighted(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    targets = targets.to(logits.device)
    mask = targets.ne(ignore_index)
    if mask.sum().item() == 0:
        return torch.zeros((), device=logits.device)
    return F.cross_entropy(logits[mask], targets[mask], weight=class_weights)


def soft_kld_loss(logits: torch.Tensor, soft_targets: torch.Tensor, has_soft: torch.Tensor) -> torch.Tensor:
    has_soft = has_soft.to(logits.device)
    soft_targets = soft_targets.to(logits.device)
    mask = has_soft.eq(1)
    if mask.sum().item() == 0:
        return torch.zeros((), device=logits.device)
    logp = F.log_softmax(logits[mask], dim=-1)
    tgt = soft_targets[mask]
    tgt = tgt / tgt.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(logp, tgt, reduction="batchmean")


def make_class_weights_from_counts(labels: pd.Series, n_classes: int) -> torch.Tensor:
    counts = labels.value_counts().sort_index()
    weights = []
    total = int(len(labels))
    for i in range(n_classes):
        c = int(counts.get(i, 1))
        weights.append(total / (n_classes * c))
    weights = np.array(weights, dtype=np.float32)
    weights = np.clip(weights, 0.5, 4.0)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def compute_tp_tn_fp_fn(y_true: List[int], y_pred: List[int], labels: List[int], names: List[str]) -> Dict[str, Dict[str, int]]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    out = {}
    for i, name in enumerate(names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        out[name] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    return out


def evaluate(model: HierMultiTaskBiasModel, dl: DataLoader) -> Dict[str, Any]:
    model.eval()
    all_lean_p, all_lean_y = [], []
    all_int_p, all_int_y = [], []
    dom_p, dom_y = [], []
    losses = []

    with torch.no_grad():
        for batch in dl:
            with torch.autocast(device_type=DEVICE, enabled=USE_AMP, dtype=AMP_DTYPE):
                out = model(batch, grl_lambda=0.0)
                l_lean = masked_ce_loss(out["logits_lean"], batch.y_lean, ignore_index=-100)
                l_int = masked_ce_loss(out["logits_int"], batch.y_int, ignore_index=-100)
                l_dom = F.cross_entropy(out["logits_dom"], batch.domain.to(DEVICE))
                loss = l_lean + l_int + W_DOMAIN * l_dom

            losses.append(float(loss.item()))

            logits_lean = out["logits_lean"].detach().float().cpu().numpy()
            for i in range(len(logits_lean)):
                y = int(batch.y_lean[i].item())
                if y == -100:
                    continue
                probs = torch.softmax(torch.tensor(logits_lean[i]), dim=-1).numpy()
                probs = apply_source_prior_if_ambiguous(probs, batch.source_name[i])
                p = int(np.argmax(probs))
                all_lean_y.append(y)
                all_lean_p.append(p)

            yI = batch.y_int.cpu().numpy()
            pI = out["logits_int"].argmax(dim=-1).detach().cpu().numpy()
            m2 = yI != -100
            if m2.any():
                all_int_y.extend(yI[m2].tolist())
                all_int_p.extend(pI[m2].tolist())

            dom_y.extend(batch.domain.cpu().numpy().tolist())
            dom_p.extend(out["logits_dom"].argmax(dim=-1).detach().cpu().numpy().tolist())

    metrics = {"eval_loss": float(np.mean(losses))}

    if all_lean_y:
        metrics["lean_acc"] = accuracy_score(all_lean_y, all_lean_p)
        metrics["lean_f1_macro"] = f1_score(all_lean_y, all_lean_p, average="macro")
        metrics["lean_confusion_stats"] = compute_tp_tn_fp_fn(all_lean_y, all_lean_p, list(range(len(LEAN_CANON))), LEAN_CANON)

    if all_int_y:
        metrics["int_acc"] = accuracy_score(all_int_y, all_int_p)
        metrics["int_f1_macro"] = f1_score(all_int_y, all_int_p, average="macro")
        metrics["int_confusion_stats"] = compute_tp_tn_fp_fn(all_int_y, all_int_p, list(range(len(INT_CANON))), INT_CANON)

    metrics["domain_acc"] = accuracy_score(dom_y, dom_p)
    return metrics


def load_lean_dataset_from_combined() -> pd.DataFrame:
    if not FILE_COMBINED_LEAN.exists():
        raise RuntimeError(f"Combined leaning dataset not found: {FILE_COMBINED_LEAN}")

    df = pd.read_csv(FILE_COMBINED_LEAN, low_memory=False)

    text_col = detect_col(df, ["text"])
    label_col = detect_col(df, ["lean", "label", "bias", "bias_rating"])
    source_col = None
    title_col = None
    link_col = None

    available_cols = {str(c).strip().lower() for c in df.columns}
    if "source_name" in available_cols:
        source_col = detect_col(df, ["source_name"])
    elif "site" in available_cols:
        source_col = detect_col(df, ["site"])
    elif "source" in available_cols:
        source_col = detect_col(df, ["source"])

    if "title" in available_cols:
        title_col = detect_col(df, ["title"])
    elif "tittle" in available_cols:
        title_col = detect_col(df, ["tittle"])

    if "link" in available_cols:
        link_col = detect_col(df, ["link"])
    elif "url" in available_cols:
        link_col = detect_col(df, ["url"])

    df[text_col] = df[text_col].apply(clean_text_value)
    df["lean"] = df[label_col].map(norm_lean)

    df = df[df["lean"].notna() & df[text_col].apply(is_valid_text)].copy()

    df["text"] = df[text_col].astype(str)
    df["title"] = df[title_col].fillna("").astype(str) if title_col is not None else ""
    df["link"] = df[link_col].fillna("").astype(str) if link_col is not None else ""
    df["source_name"] = df[source_col].fillna("").astype(str) if source_col is not None else ""
    df["dataset_name"] = df["dataset_name"].fillna("combined_lean_train_without_holdout").astype(str) if "dataset_name" in df.columns else "combined_lean_train_without_holdout"

    if "is_synthetic" in df.columns:
        synth_counts = df["is_synthetic"].value_counts(dropna=False).to_dict()
        print(f"[combined_lean_train_without_holdout] synthetic flag counts: {synth_counts}")

    df["row_id"] = np.arange(len(df))
    df["y_lean"] = df["lean"].map(LEAN_TO_ID).astype(int)
    # Derive intensity from lean: extreme lean → Highly Biased, moderate → Slightly Biased, center → Neutral.
    # This ensures the intensity head trains on article-length text (same distribution as inference),
    # not only on the 13-word social-media snippets from newsmediabias.
    _LEAN_TO_INT = {
        LEAN_TO_ID["Right"]:        INT_TO_ID["Highly Biased"],
        LEAN_TO_ID["Left"]:         INT_TO_ID["Highly Biased"],
        LEAN_TO_ID["Right-center"]: INT_TO_ID["Slightly Biased"],
        LEAN_TO_ID["Left-center"]:  INT_TO_ID["Slightly Biased"],
        LEAN_TO_ID["Center"]:       INT_TO_ID["Neutral"],
    }
    df["y_int"] = df["y_lean"].map(_LEAN_TO_INT).astype(int)
    df["domain"] = DOMAIN_LEAN
    df["lean_soft"] = None

    print(f"[combined_lean_train_without_holdout] usable rows: {len(df)}")
    print(df["lean"].value_counts(dropna=False))
    for i, name in enumerate(LEAN_CANON):
        cnt = int((df["y_lean"] == i).sum())
        print(f"  {name}: {cnt}")

    return df[[
        "row_id", "title", "link", "text", "source_name",
        "dataset_name", "y_lean", "y_int", "domain", "lean_soft"
    ]]


def load_intensity_dataset() -> pd.DataFrame:
    if not FILE_NEWSMEDIABIAS.exists():
        raise RuntimeError("newsmediabias-full.csv not found.")

    try:
        df = pd.read_csv(FILE_NEWSMEDIABIAS, engine="python", on_bad_lines="skip", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(FILE_NEWSMEDIABIAS, engine="python", on_bad_lines="skip", encoding="latin1")

    print(f"[newsmediabias] loaded raw rows: {len(df)}")

    text_col = detect_col(df, ["text"])
    label_col = detect_col(df, ["label", "bias", "bias_rating"])

    source_col = None
    available_cols = {str(c).strip().lower() for c in df.columns}
    for cand in ["source_name", "source", "name"]:
        if cand in available_cols:
            source_col = detect_col(df, [cand])
            break

    df[text_col] = df[text_col].apply(clean_text_value)
    df["intensity"] = df[label_col].map(norm_intensity)
    df = df[df["intensity"].notna() & df[text_col].apply(is_valid_text)].copy()
    df = df.rename(columns={text_col: "text"})

    if source_col is None:
        df["source_name"] = ""
    else:
        df = df.rename(columns={source_col: "source_name"})

    # Cap per class so 3.4 M sentence-level rows don't overwhelm the 74.5 K
    # article-level intensity labels derived from the lean dataset.
    df = (
        df.groupby("intensity", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), NEWSMEDIABIAS_MAX_PER_CLASS), random_state=42))
        .reset_index(drop=True)
    )
    print(f"[newsmediabias] after per-class cap ({NEWSMEDIABIAS_MAX_PER_CLASS}/class):")
    print(df["intensity"].value_counts(dropna=False))

    df["title"] = ""
    df["link"] = ""
    df["dataset_name"] = "newsmediabias"
    df["row_id"] = np.arange(len(df))
    df["y_int"] = df["intensity"].map(INT_TO_ID).astype(int)
    df["y_lean"] = -100
    df["domain"] = DOMAIN_INTENSITY
    df["lean_soft"] = None

    print(f"[newsmediabias] usable rows: {len(df)}")

    return df[[
        "row_id", "title", "link", "text", "source_name",
        "dataset_name", "y_lean", "y_int", "domain", "lean_soft"
    ]]


def load_and_prepare_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return load_lean_dataset_from_combined(), load_intensity_dataset()


def split_70_15_15(df: pd.DataFrame, stratify_col: str, seed: int = 42):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df[stratify_col]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df[stratify_col]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def clear_pseudo_cache():
    if PSEUDO_CACHE.exists():
        PSEUDO_CACHE.unlink()
    if PSEUDO_CACHE_CSV.exists():
        PSEUDO_CACHE_CSV.unlink()
    if PSEUDO_PARTS_DIR.exists():
        for p in PSEUDO_PARTS_DIR.glob("*"):
            p.unlink()
        PSEUDO_PARTS_DIR.rmdir()


def _write_pseudo_chunk(df_chunk: pd.DataFrame, first_write: bool):
    try:
        if first_write:
            df_chunk.to_parquet(PSEUDO_CACHE, index=False)
        else:
            PSEUDO_PARTS_DIR.mkdir(exist_ok=True)
            part_path = PSEUDO_PARTS_DIR / f"part_{random.randint(0, 10**12)}.parquet"
            df_chunk.to_parquet(part_path, index=False)
    except Exception:
        mode = "wt" if first_write else "at"
        header = first_write
        df_chunk.to_csv(PSEUDO_CACHE_CSV, index=False, mode=mode, header=header, compression="gzip")


def _read_pseudo_cache() -> pd.DataFrame:
    frames = []

    if PSEUDO_CACHE.exists():
        try:
            frames.append(pd.read_parquet(PSEUDO_CACHE))
        except Exception:
            pass

    if PSEUDO_PARTS_DIR.exists():
        for p in sorted(PSEUDO_PARTS_DIR.glob("part_*.parquet")):
            try:
                frames.append(pd.read_parquet(p))
            except Exception:
                pass

    if PSEUDO_CACHE_CSV.exists():
        try:
            frames.append(pd.read_csv(PSEUDO_CACHE_CSV))
        except Exception:
            pass

    if not frames:
        raise RuntimeError("Pseudo-label cache not found. Run --pseudo_label_articles first.")

    df = pd.concat(frames, ignore_index=True)
    if "row_id" in df.columns:
        df = df.drop_duplicates(subset=["row_id"], keep="last").reset_index(drop=True)
    return df


def soften_center_adjacent(probs: np.ndarray) -> np.ndarray:
    p = probs.copy()

    idx_center = LEAN_TO_ID["Center"]
    idx_rc = LEAN_TO_ID["Right-center"]
    idx_lc = LEAN_TO_ID["Left-center"]

    center = p[idx_center]
    rc = p[idx_rc]
    lc = p[idx_lc]

    if center >= 0.30 and rc >= 0.20 and abs(center - rc) <= 0.18:
        p[idx_center] *= 0.90
        p[idx_rc] *= 1.10

    if center >= 0.30 and lc >= 0.20 and abs(center - lc) <= 0.18:
        p[idx_center] *= 0.90
        p[idx_lc] *= 1.10

    idx_r = LEAN_TO_ID["Right"]
    idx_l = LEAN_TO_ID["Left"]

    if p[idx_r] >= 0.25 and p[idx_rc] >= 0.20:
        p[idx_rc] *= 1.05
    if p[idx_l] >= 0.25 and p[idx_lc] >= 0.20:
        p[idx_lc] *= 1.05

    p = p / p.sum()
    return p


def train_teacher(seed: int, encoder_path: str, use_compile: bool = False) -> str:
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(encoder_path)

    df_lean, _ = load_and_prepare_raw()
    train_df, val_df, test_df = split_70_15_15(df_lean, "y_lean", seed=seed)

    print("\n[Teacher] train split distribution:")
    for i, name in enumerate(LEAN_CANON):
        cnt = int((train_df["y_lean"] == i).sum())
        print(f"  {name}: {cnt}")

    train_ds = StudentTextDataset(train_df)
    val_ds = StudentTextDataset(val_df)
    test_ds = StudentTextDataset(test_df)

    collator = StudentHierTextCollator(tok, MAX_LENGTH, MAX_CHUNKS)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), collate_fn=collator)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE == "cuda"), collate_fn=collator)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE == "cuda"), collate_fn=collator)

    model = HierMultiTaskBiasModel(
        encoder_name=encoder_path,
        feat_dim=14,
        n_lean=len(LEAN_CANON),
        n_int=len(INT_CANON),
        n_domain=2,
    ).to(DEVICE)

    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    updates_per_epoch = math.ceil(len(train_dl) / GRAD_ACCUM)
    total_updates = EPOCHS_TEACHER * updates_per_epoch

    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(total_updates * WARMUP_RATIO),
        num_training_steps=total_updates,
    )

    scaler = make_grad_scaler()

    print("[Teacher] training...")
    global_step = 0

    for epoch in range(1, EPOCHS_TEACHER + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        pbar = tqdm(
            enumerate(train_dl, start=1),
            total=len(train_dl),
            dynamic_ncols=True,
            desc=f"teacher epoch {epoch}/{EPOCHS_TEACHER}"
        )

        for batch_idx, batch in pbar:
            with torch.autocast(device_type=DEVICE, enabled=USE_AMP, dtype=AMP_DTYPE):
                out = model(batch, grl_lambda=0.0)
                loss = masked_ce_loss(out["logits_lean"], batch.y_lean, ignore_index=-100)
                loss = loss / GRAD_ACCUM

            if DEVICE == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_update = (batch_idx % GRAD_ACCUM == 0) or (batch_idx == len(train_dl))
            if is_update:
                if DEVICE == "cuda":
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

            pbar.set_postfix({"loss": f"{(loss.item() * GRAD_ACCUM):.3f}", "upd": global_step})

        metrics = evaluate(model, val_dl)
        print(f"[Teacher eval][epoch {epoch}] {json.dumps(metrics, indent=2)}")

    test_metrics = evaluate(model, test_dl)
    print("[Teacher] final TEST metrics:")
    print(json.dumps(test_metrics, indent=2))

    (OUT_DIR / "teacher_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    save_dir = OUT_DIR / "teacher_lean_v3"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")
    tok.save_pretrained(save_dir)
    (save_dir / "meta.json").write_text(json.dumps({"seed": seed, "encoder_path": encoder_path}, indent=2), encoding="utf-8")
    return str(save_dir)


def load_model(model_dir: str, encoder_path: str) -> Tuple[Any, HierMultiTaskBiasModel]:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = HierMultiTaskBiasModel(
        encoder_name=encoder_path,
        feat_dim=14,
        n_lean=5,
        n_int=3,
        n_domain=2
    ).to(DEVICE)
    sd = torch.load(Path(model_dir) / "model.pt", map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return tok, model


@torch.no_grad()
def pseudo_label_articles(teacher_dir: str, encoder_path: str, infer_batch: int = INFER_BATCH, use_compile: bool = False):
    if PSEUDO_CACHE.exists() or PSEUDO_PARTS_DIR.exists() or PSEUDO_CACHE_CSV.exists():
        print("[Pseudo-label] cache already exists. Skipping.")
        return

    _, df_int = load_and_prepare_raw()
    tok, model = load_model(teacher_dir, encoder_path)

    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    ds = PseudoTextDataset(df_int)
    collator = HierTextCollator(tok, MAX_LENGTH, MAX_CHUNKS)
    dl = DataLoader(ds, batch_size=infer_batch, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), collate_fn=collator)

    print(f"[Pseudo-label] generating lean soft labels for {len(ds):,} intensity samples...")

    first_write = True
    buffer_rows = []
    flush_every = 20000
    kept_soft = 0

    for item in tqdm(dl, desc="pseudo-label", dynamic_ncols=True):
        row_ids = item["row_id"].cpu().numpy().astype(int).tolist()
        source_names = item["source_name"]
        batch: Batch = item["batch"]

        with torch.autocast(device_type=DEVICE, enabled=USE_AMP, dtype=AMP_DTYPE):
            out = model(batch, grl_lambda=0.0)
            raw_probs = torch.softmax(out["logits_lean"], dim=-1).float().cpu().numpy()

        for rid, src, p in zip(row_ids, source_names, raw_probs):
            p = apply_source_prior_if_ambiguous(p, src)
            p = soften_center_adjacent(p)

            top_idx = int(np.argmax(p))
            top_name = LEAN_CANON[top_idx]
            top_conf = float(np.max(p))

            keep_soft = (top_conf >= PSEUDO_MIN_CONF) and (top_name != "Center")
            if keep_soft:
                kept_soft += 1
                row = {
                    "row_id": rid,
                    "p0": float(p[0]),
                    "p1": float(p[1]),
                    "p2": float(p[2]),
                    "p3": float(p[3]),
                    "p4": float(p[4]),
                }
            else:
                row = {
                    "row_id": rid,
                    "p0": np.nan,
                    "p1": np.nan,
                    "p2": np.nan,
                    "p3": np.nan,
                    "p4": np.nan,
                }

            buffer_rows.append(row)

        if len(buffer_rows) >= flush_every:
            df_chunk = pd.DataFrame(buffer_rows)
            _write_pseudo_chunk(df_chunk, first_write=first_write)
            first_write = False
            buffer_rows = []

    if buffer_rows:
        df_chunk = pd.DataFrame(buffer_rows)
        _write_pseudo_chunk(df_chunk, first_write=first_write)

    print(f"[Pseudo-label] kept soft labels: {kept_soft}/{len(ds)} ({100.0 * kept_soft / max(len(ds), 1):.2f}%)")
    print("[Pseudo-label] done.")


def build_student_dataframes(seed: int = 42):
    df_lean, df_int = load_and_prepare_raw()
    pseudo_df = _read_pseudo_cache()

    df_int = df_int.merge(pseudo_df, on="row_id", how="left")

    def make_soft(row):
        if pd.isna(row["p0"]):
            return None
        probs = [float(row["p0"]), float(row["p1"]), float(row["p2"]), float(row["p3"]), float(row["p4"])]
        top_idx = int(np.argmax(probs))
        top_name = LEAN_CANON[top_idx]
        if top_name == "Center":
            return None
        return probs

    df_int["lean_soft"] = df_int.apply(make_soft, axis=1)

    for c in ["p0", "p1", "p2", "p3", "p4"]:
        if c in df_int.columns:
            df_int.drop(columns=[c], inplace=True)

    n_soft = int(df_int["lean_soft"].notna().sum())
    n_all = int(len(df_int))
    print(f"[Student] intensity rows with soft lean labels: {n_soft}/{n_all} ({100.0 * n_soft / max(n_all, 1):.2f}%)")

    lean_train, lean_val, lean_test = split_70_15_15(df_lean, "y_lean", seed=seed)
    int_train, int_val, int_test = split_70_15_15(df_int, "y_int", seed=seed)

    print(f"Lean train/val/test: {len(lean_train)} / {len(lean_val)} / {len(lean_test)}")
    print(f"Intensity train/val/test: {len(int_train)} / {len(int_val)} / {len(int_test)}")

    train_df = pd.concat([lean_train, int_train], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat([lean_val, int_val], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat([lean_test, int_test], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


def maybe_take_training_break(start_time: float, save_callback=None):
    if not ENABLE_TRAINING_BREAKS:
        return start_time
    elapsed_hours = (time.time() - start_time) / 3600.0
    if elapsed_hours >= BREAK_EVERY_HOURS:
        print("\nCooling break started")
        print("Time:", datetime.now().strftime("%H:%M:%S"))
        if save_callback is not None:
            save_callback()
        sleep_seconds = BREAK_DURATION_MIN * 60
        for remaining in range(sleep_seconds, 0, -60):
            print(f"Cooling... {remaining // 60} min remaining")
            time.sleep(60)
        print("Break finished. Resuming training.\n")
        return time.time()
    return start_time


def train_student(seed: int, encoder_path: str, run_name: str, use_compile: bool = False) -> str:
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(encoder_path)

    print("[Student] building train/val/test...")
    train_df, val_df, test_df = build_student_dataframes(seed=seed)

    lean_train_only = train_df[train_df["y_lean"] != -100].copy()
    int_train_only = train_df[train_df["y_int"] != -100].copy()

    lean_class_weights = make_class_weights_from_counts(lean_train_only["y_lean"], len(LEAN_CANON))
    int_class_weights = make_class_weights_from_counts(int_train_only["y_int"], len(INT_CANON))

    print("[Student] lean class weights:", lean_class_weights.detach().cpu().tolist())
    print("[Student] int class weights:", int_class_weights.detach().cpu().tolist())

    train_ds = StudentTextDataset(train_df)
    val_ds = StudentTextDataset(val_df)
    test_ds = StudentTextDataset(test_df)

    collator = StudentHierTextCollator(tok, MAX_LENGTH, MAX_CHUNKS)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), collate_fn=collator)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE == "cuda"), collate_fn=collator)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE == "cuda"), collate_fn=collator)

    model = HierMultiTaskBiasModel(
        encoder_name=encoder_path,
        feat_dim=14,
        n_lean=len(LEAN_CANON),
        n_int=len(INT_CANON),
        n_domain=2,
    ).to(DEVICE)

    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    updates_per_epoch = math.ceil(len(train_dl) / GRAD_ACCUM)
    total_updates = EPOCHS_STUDENT * updates_per_epoch

    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(total_updates * WARMUP_RATIO),
        num_training_steps=total_updates,
    )

    scaler = make_grad_scaler()

    print("[Student] multitask training...")
    update_step = 0
    training_start_time = time.time()
    freeze_first_epoch = False

    def save_checkpoint():
        torch.save(model.state_dict(), OUT_DIR / "student_checkpoint.pt")

    for epoch in range(1, EPOCHS_STUDENT + 1):
        for p in model.encoder.parameters():
            p.requires_grad = not (freeze_first_epoch and epoch == 1)

        model.train()
        opt.zero_grad(set_to_none=True)

        pbar = tqdm(
            enumerate(train_dl, start=1),
            total=len(train_dl),
            dynamic_ncols=True,
            desc=f"student epoch {epoch}/{EPOCHS_STUDENT}"
        )

        for batch_idx, batch in pbar:
            training_start_time = maybe_take_training_break(training_start_time, save_callback=save_checkpoint)
            progress = update_step / max(total_updates, 1)
            grl_lambda = float(2.0 / (1.0 + math.exp(-10 * progress)) - 1.0)

            with torch.autocast(device_type=DEVICE, enabled=USE_AMP, dtype=AMP_DTYPE):
                out = model(batch, grl_lambda=grl_lambda)

                l_lean_hard = masked_ce_loss_weighted(
                    out["logits_lean"],
                    batch.y_lean,
                    class_weights=lean_class_weights,
                    ignore_index=-100,
                )
                l_lean_soft = soft_kld_loss(out["logits_lean"], batch.lean_soft, batch.has_lean_soft)
                l_int = masked_ce_loss_weighted(
                    out["logits_int"],
                    batch.y_int,
                    class_weights=int_class_weights,
                    ignore_index=-100,
                )
                l_dom = F.cross_entropy(out["logits_dom"], batch.domain.to(DEVICE))

                loss = (
                    (W_LEAN_HARD * l_lean_hard) +
                    (W_LEAN_SOFT * l_lean_soft) +
                    (W_INTENSITY * l_int) +
                    (W_DOMAIN * l_dom)
                )
                loss = loss / GRAD_ACCUM

            if DEVICE == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_update = (batch_idx % GRAD_ACCUM == 0) or (batch_idx == len(train_dl))
            if is_update:
                if DEVICE == "cuda":
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                opt.zero_grad(set_to_none=True)
                sched.step()
                update_step += 1

            pbar.set_postfix({
                "loss": f"{(loss.item() * GRAD_ACCUM):.3f}",
                "upd": update_step,
                "grl": f"{grl_lambda:.2f}",
            })

        metrics = evaluate(model, val_dl)
        print(f"[Student eval][epoch {epoch}] {json.dumps(metrics, indent=2)}")

    test_metrics = evaluate(model, test_dl)
    print("[Student] final TEST metrics:")
    print(json.dumps(test_metrics, indent=2))

    (OUT_DIR / "student_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    save_dir = OUT_DIR / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")
    tok.save_pretrained(save_dir)
    (save_dir / "meta.json").write_text(json.dumps({"seed": seed, "encoder_path": encoder_path}, indent=2), encoding="utf-8")
    return str(save_dir)


@torch.no_grad()
def _predict_from_model(
    text: str,
    model_dir: str,
    encoder_path: str,
    source_name: Optional[str] = None,
):
    tok, model = load_model(model_dir, encoder_path)
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
        source_name=[source_name or ""],
    )

    with torch.autocast(device_type=DEVICE, enabled=USE_AMP, dtype=AMP_DTYPE):
        out = model(batch, grl_lambda=0.0)

    probs_lean = torch.softmax(out["logits_lean"], dim=-1).float().squeeze(0).cpu().numpy()
    probs_lean = apply_source_prior_if_ambiguous(probs_lean, source_name)
    probs_int = torch.softmax(out["logits_int"], dim=-1).float().squeeze(0).cpu().numpy()
    chunk_attn = out["chunk_attn"].float().squeeze(0).cpu().numpy()

    return {
        "probs_lean": probs_lean,
        "probs_int": probs_int,
        "chunk_attention": chunk_attn.tolist(),
    }


@torch.no_grad()
def predict(
    text: str,
    model_dir: str,
    encoder_path: str,
    threshold_non_center: float = 0.5,
    source_name: Optional[str] = None,
):
    raw = _predict_from_model(text, model_dir, encoder_path, source_name=source_name)

    probs_lean = raw["probs_lean"]
    probs_int = raw["probs_int"]

    pred_lean = LEAN_CANON[int(np.argmax(probs_lean))]

    p_center = float(probs_lean[LEAN_TO_ID["Center"]])
    biased_score = 1.0 - p_center
    biased = biased_score >= threshold_non_center

    # Intensity derived from biased_score (calibrated on the real holdout set).
    # The trained intensity head and lean-label mapping both fail on OOD articles
    # because the lean model rarely assigns probability to moderate classes on
    # unseen sources.  biased_score separates Center (median 0.43) from biased
    # articles reliably and produces a realistic 17/33/50 Neutral/Slight/High split.
    if biased_score < INTENSITY_THRESH_NEUTRAL:
        pred_int = "Neutral"
    elif biased_score < INTENSITY_THRESH_SLIGHTLY:
        pred_int = "Slightly Biased"
    else:
        pred_int = "Highly Biased"

    return {
        "political_bias": pred_lean,
        "bias_intensity": pred_int,
        "biased": biased,
        "biased_score": float(biased_score),
        "probs_lean": {LEAN_CANON[i]: float(probs_lean[i]) for i in range(5)},
        "probs_int": {INT_CANON[i]: float(probs_int[i]) for i in range(3)},
        "chunk_attention": raw["chunk_attention"],
        "source_used": source_name or None,
    }


@torch.no_grad()
def predict_combined(
    text: str,
    teacher_dir: str,
    student_dir: str,
    encoder_path: str,
    threshold_non_center: float = 0.5,
    source_name: Optional[str] = None,
):
    teacher_raw = _predict_from_model(text, teacher_dir, encoder_path, source_name=source_name)
    student_raw = _predict_from_model(text, student_dir, encoder_path, source_name=source_name)

    probs_lean = teacher_raw["probs_lean"]
    probs_int = student_raw["probs_int"]

    pred_lean = LEAN_CANON[int(np.argmax(probs_lean))]
    p_center = float(probs_lean[LEAN_TO_ID["Center"]])
    biased_score = 1.0 - p_center
    biased = biased_score >= threshold_non_center

    if biased_score < INTENSITY_THRESH_NEUTRAL:
        pred_int = "Neutral"
    elif biased_score < INTENSITY_THRESH_SLIGHTLY:
        pred_int = "Slightly Biased"
    else:
        pred_int = "Highly Biased"

    return {
        "political_bias": pred_lean,
        "bias_intensity": pred_int,
        "biased": biased,
        "biased_score": float(biased_score),
        "probs_lean": {LEAN_CANON[i]: float(probs_lean[i]) for i in range(5)},
        "probs_int": {INT_CANON[i]: float(probs_int[i]) for i in range(3)},
        "chunk_attention_teacher": teacher_raw["chunk_attention"],
        "chunk_attention_student": student_raw["chunk_attention"],
        "source_used": source_name or None,
        "prediction_mode": "combined_teacher_lean_student_intensity",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=str, default=BASE_MODEL)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--train_teacher", action="store_true")
    ap.add_argument("--pseudo_label_articles", action="store_true")
    ap.add_argument("--train_student", action="store_true")
    ap.add_argument("--clear_pseudo_cache", action="store_true")

    ap.add_argument("--infer_batch", type=int, default=INFER_BATCH)
    ap.add_argument("--compile", action="store_true")

    ap.add_argument("--predict", type=str, default=None)
    ap.add_argument("--predict_source", type=str, default=None)
    ap.add_argument("--predict_mode", type=str, default="combined", choices=["teacher", "student", "combined"])
    ap.add_argument("--threshold", type=float, default=0.5)

    args = ap.parse_args()

    print("Device:", DEVICE)
    encoder_path = args.encoder
    teacher_dir = str(OUT_DIR / "teacher_lean_v3")

    if args.clear_pseudo_cache:
        clear_pseudo_cache()
        print("[Pseudo-label] cache cleared.")

    if args.train_teacher:
        teacher_dir = train_teacher(seed=args.seed, encoder_path=encoder_path, use_compile=args.compile)

    if args.pseudo_label_articles:
        if not Path(teacher_dir).exists():
            raise RuntimeError("Teacher model not found. Run --train_teacher first.")
        pseudo_label_articles(
            teacher_dir=teacher_dir,
            encoder_path=encoder_path,
            infer_batch=args.infer_batch,
            use_compile=args.compile
        )

    if args.train_student:
        run_name = f"student_mt_softlean_v3_seed{args.seed}"
        student_dir = train_student(
            seed=args.seed,
            encoder_path=encoder_path,
            run_name=run_name,
            use_compile=args.compile
        )
        (OUT_DIR / "last_student.json").write_text(json.dumps({"student_dir": student_dir}, indent=2), encoding="utf-8")

    if args.predict is not None:
        if args.predict_mode == "teacher":
            if not Path(teacher_dir).exists():
                raise RuntimeError("Teacher model not found. Run --train_teacher first.")
            res = predict(
                args.predict,
                model_dir=teacher_dir,
                encoder_path=encoder_path,
                threshold_non_center=args.threshold,
                source_name=args.predict_source,
            )
            res["prediction_mode"] = "teacher"
            print(json.dumps(res, indent=2))
            return

        last = OUT_DIR / "last_student.json"
        if not last.exists():
            raise RuntimeError("No trained student found. Run --train_student first.")
        student_dir = json.loads(last.read_text(encoding="utf-8"))["student_dir"]

        if args.predict_mode == "student":
            res = predict(
                args.predict,
                model_dir=student_dir,
                encoder_path=encoder_path,
                threshold_non_center=args.threshold,
                source_name=args.predict_source,
            )
            res["prediction_mode"] = "student"
            print(json.dumps(res, indent=2))
            return

        if not Path(teacher_dir).exists():
            raise RuntimeError("Teacher model not found. Run --train_teacher first.")

        res = predict_combined(
            args.predict,
            teacher_dir=teacher_dir,
            student_dir=student_dir,
            encoder_path=encoder_path,
            threshold_non_center=args.threshold,
            source_name=args.predict_source,
        )
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()