import os
import re
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from transformers import AutoTokenizer, AutoModel

# ============================================================
# DEVICE
# ============================================================
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

USE_AMP = DEVICE in {"cuda", "mps"}
AMP_DTYPE = torch.float16 if DEVICE in {"cuda", "mps"} else None

# ============================================================
# LABELS
# ============================================================
# model order from your v3 code
LEAN_CANON = ["Right", "Right-center", "Center", "Left-center", "Left"]

# dataset order:
# 0-left, 1-lean left, 2-center, 3-lean right, 4-right
DATASET_ID_TO_MODEL_ID = {
    0: 4,  # Left
    1: 3,  # Left-center
    2: 2,  # Center
    3: 1,  # Right-center
    4: 0,  # Right
}

MODEL_ID_TO_DATASET_ID = {v: k for k, v in DATASET_ID_TO_MODEL_ID.items()}

# ============================================================
# TEXT FEATURES
# ============================================================
TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)

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
    "huge", "massive", "disaster", "outrage",
    "awfully", "extraordinary", "unusual", "much", "rather", "entirely",
    "greatly", "really", "exceedingly", "too", "completely", "terribly",
    "perfectly", "quite", "certainly", "especially", "fairly", "highly",
    "increasingly", "much more", "particularly", "probably", "more",
    "absolutely", "intensely", "supremely", "most", "pretty"
}

NEGATIONS = {
    "no", "not", "none", "never", "neither", "nobody", "nothing", "nowhere",
    "seldom", "scarcely", "hardly", "barely", "is not", "cannot", "may not",
    "could not", "would not", "did not", "do not", "does not", "was not",
    "are not", "were not"
}


def normalize_term(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


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

# ============================================================
# CHUNKING
# ============================================================
def encode_ids_to_chunks(ids, cls_id, sep_id, pad_id, max_length, max_chunks):
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

# ============================================================
# MODEL
# ============================================================
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

    def forward(self, batch: Batch, grl_lambda: float = 0.0):
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

# ============================================================
# DATASET
# ============================================================
class EvalDataset(TorchDataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, max_chunks: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_chunks = max_chunks
        self.cls_id = int(tokenizer.cls_token_id)
        self.sep_id = int(tokenizer.sep_token_id)
        self.pad_id = int(tokenizer.pad_token_id)
        self.max_total = (max_length - 2) * max_chunks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        y_true = int(row["y_true_model"])

        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_total,
            return_attention_mask=False,
        )
        ids = enc["input_ids"]
        padded, am, cm = encode_ids_to_chunks(
            ids,
            self.cls_id,
            self.sep_id,
            self.pad_id,
            self.max_length,
            self.max_chunks,
        )

        return {
            "input_ids": padded,
            "attention_mask": am,
            "chunk_mask": cm,
            "feats": extract_features(text),
            "y_true": y_true,
            "text": text,
        }


def collate_eval(examples):
    B = len(examples)
    C = len(examples[0]["chunk_mask"])
    L = len(examples[0]["input_ids"][0])

    input_ids = torch.zeros((B, C, L), dtype=torch.long)
    attention_mask = torch.zeros((B, C, L), dtype=torch.long)
    chunk_mask = torch.zeros((B, C), dtype=torch.long)
    feats = torch.zeros((B, 14), dtype=torch.float32)
    y_lean = torch.zeros((B,), dtype=torch.long)

    for i, ex in enumerate(examples):
        input_ids[i] = torch.tensor(ex["input_ids"], dtype=torch.long)
        attention_mask[i] = torch.tensor(ex["attention_mask"], dtype=torch.long)
        chunk_mask[i] = torch.tensor(ex["chunk_mask"], dtype=torch.long)
        feats[i] = torch.tensor(ex["feats"], dtype=torch.float32)
        y_lean[i] = int(ex["y_true"])

    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        chunk_mask=chunk_mask,
        feats=feats,
        y_lean=y_lean,
        y_int=torch.full((B,), -100, dtype=torch.long),
        domain=torch.zeros((B,), dtype=torch.long),
        lean_soft=torch.zeros((B, 5), dtype=torch.float32),
        has_lean_soft=torch.zeros((B,), dtype=torch.long),
        source_name=[""] * B,
    )

# ============================================================
# LOADING
# ============================================================
def load_model(model_dir: str, encoder_name: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = HierMultiTaskBiasModel(
        encoder_name=encoder_name,
        feat_dim=14,
        n_lean=5,
        n_int=3,
        n_domain=2,
    ).to(DEVICE)

    sd = torch.load(Path(model_dir) / "model.pt", map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return tok, model


def load_eval_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain columns: text, label")

    df = df.copy()
    df["text"] = df["text"].astype(str).fillna("")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    valid_labels = set(DATASET_ID_TO_MODEL_ID.keys())
    df = df[df["label"].isin(valid_labels)].copy()

    df["y_true_model"] = df["label"].map(DATASET_ID_TO_MODEL_ID).astype(int)
    df = df.reset_index(drop=True)
    return df

# ============================================================
# EVAL
# ============================================================
@torch.no_grad()
def evaluate_dataset(
    csv_path: str,
    model_dir: str,
    encoder_name: str,
    max_length: int,
    max_chunks: int,
    batch_size: int,
    out_png: str,
):
    df = load_eval_dataset(csv_path)
    print(f"Loaded rows: {len(df)}")

    tok, model = load_model(model_dir, encoder_name)

    ds = EvalDataset(df, tok, max_length=max_length, max_chunks=max_chunks)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_eval,
    )

    y_true = []
    y_pred = []

    for batch in tqdm(dl, desc="evaluating", dynamic_ncols=True):
        with torch.autocast(device_type=DEVICE, enabled=USE_AMP, dtype=AMP_DTYPE):
            out = model(batch, grl_lambda=0.0)

        preds = out["logits_lean"].argmax(dim=-1).detach().cpu().numpy()
        true = batch.y_lean.detach().cpu().numpy()

        y_true.extend(true.tolist())
        y_pred.extend(preds.tolist())

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    print("\n=== RESULTS ===")
    print(f"Accuracy: {acc:.6f}")
    print(f"Macro F1: {f1m:.6f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            target_names=LEAN_CANON,
            digits=4,
            zero_division=0,
        )
    )

    print("\nConfusion matrix:")
    print(cm)

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(LEAN_CANON)),
        yticks=np.arange(len(LEAN_CANON)),
        xticklabels=LEAN_CANON,
        yticklabels=LEAN_CANON,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved confusion matrix plot to: {out_png}")

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(f1m),
        "confusion_matrix": cm.tolist(),
        "labels_order": LEAN_CANON,
    }

    out_json = str(Path(out_png).with_suffix(".json"))
    Path(out_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics json to: {out_json}")


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to evaluation CSV")
    ap.add_argument("--model_dir", type=str, required=True, help="Path to trained model dir")
    ap.add_argument("--encoder", type=str, default="bert-base-uncased")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_chunks", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_png", type=str, default="confusion_matrix_political_bias.png")
    args = ap.parse_args()

    print("Device:", DEVICE)

    evaluate_dataset(
        csv_path=args.csv,
        model_dir=args.model_dir,
        encoder_name=args.encoder,
        max_length=args.max_length,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size,
        out_png=args.out_png,
    )


if __name__ == "__main__":
    main()