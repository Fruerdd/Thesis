"""
train_bert_baseline.py

Vanilla BERT baseline for 5-class political leaning.
No chunking, no handcrafted features, no multi-task, no GRL, no distillation.
Just bert-base-uncased + one linear head + cross-entropy.

Truncates each article to 512 tokens (standard BERT max).
Trains on data_prepared/combined_lean_dataset_balanced_15000.csv (75k rows),
evaluates on data_prepared/combined_lean_real_holdout_5class_100_each.csv (500 rows).

Usage:
    python train_bert_baseline.py
    python train_bert_baseline.py --epochs 3 --batch_size 16
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ── config ────────────────────────────────────────────────────────────────────

ENCODER      = "bert-base-uncased"
MAX_LENGTH   = 512
BATCH_SIZE   = 8
EPOCHS       = 2
LR           = 2e-5
WARMUP_RATIO = 0.06
SEED         = 42

TRAIN_CSV   = "data_prepared/combined_lean_dataset_balanced_15000.csv"
HOLDOUT_CSV = "data_prepared/combined_lean_real_holdout_5class_100_each.csv"
OUT_DIR     = Path("bert_baseline_output")

LEAN_CANON = ["Right", "Right-center", "Center", "Left-center", "Left"]
LABEL2ID   = {c: i for i, c in enumerate(LEAN_CANON)}
ID2LABEL   = {i: c for i, c in enumerate(LEAN_CANON)}

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

BAD_TEXT = {"", "null", "none", "nan", "error fetching article", "<null>", "n/a"}


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def clean(x) -> str:
    s = str(x).strip() if x is not None else ""
    return "" if s.lower() in BAD_TEXT else s


def is_valid(s: str) -> bool:
    return bool(s) and len(s) >= 30 and sum(c.isalpha() for c in s) >= 15


# ── dataset ───────────────────────────────────────────────────────────────────

class LeanDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── data loading ──────────────────────────────────────────────────────────────

def load_train_val(csv_path: str, val_frac: float = 0.15, seed: int = SEED):
    df = pd.read_csv(csv_path, low_memory=False)
    df["_text"]  = df["text"].apply(clean)
    df["_label"] = df["lean"].astype(str).str.strip()
    df = df[df["_text"].apply(is_valid) & df["_label"].isin(LEAN_CANON)].copy()
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    cut = int(len(df) * (1 - val_frac))
    train_df = df.iloc[:cut]
    val_df   = df.iloc[cut:]

    print(f"Train: {len(train_df)}  Val: {len(val_df)}")
    print("Train distribution:")
    print(train_df["_label"].value_counts().to_string())

    train_texts  = train_df["_text"].tolist()
    train_labels = [LABEL2ID[l] for l in train_df["_label"]]
    val_texts    = val_df["_text"].tolist()
    val_labels   = [LABEL2ID[l] for l in val_df["_label"]]
    return train_texts, train_labels, val_texts, val_labels


def load_holdout(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)
    df["_text"]  = df["text"].apply(clean)
    df["_label"] = df["label"].astype(str).str.strip()
    df = df[df["_text"].apply(is_valid) & df["_label"].isin(LEAN_CANON)].reset_index(drop=True)
    print(f"Holdout: {len(df)} rows")
    print(df["_label"].value_counts().to_string())
    texts  = df["_text"].tolist()
    labels = [LABEL2ID[l] for l in df["_label"]]
    return texts, labels


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"]
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = out.logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    acc = accuracy_score(all_labels, all_preds)
    mf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, mf1, all_preds, all_labels


# ── confusion matrix plot ─────────────────────────────────────────────────────

def plot_cm(cm: np.ndarray, labels: list[str], out_path: str, title: str, normalize: bool = False):
    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums != 0)

    light_blues = LinearSegmentedColormap.from_list(
        "light_blues", plt.cm.Blues(np.linspace(0.05, 0.65, 256))
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, interpolation="nearest", cmap=light_blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else str(int(v)),
                    ha="center", va="center", color="black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── training loop ─────────────────────────────────────────────────────────────

def train(args):
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Encoder: {ENCODER}")
    print(f"Max len: {MAX_LENGTH}\n")

    tok = AutoTokenizer.from_pretrained(ENCODER)

    print("Loading training data...")
    train_texts, train_labels, val_texts, val_labels = load_train_val(TRAIN_CSV)
    print("\nLoading holdout data...")
    holdout_texts, holdout_labels = load_holdout(HOLDOUT_CSV)

    print("\nTokenising (this takes a moment for 75k articles at 512 tokens)...")
    train_ds   = LeanDataset(train_texts,   train_labels,   tok, MAX_LENGTH)
    val_ds     = LeanDataset(val_texts,     val_labels,     tok, MAX_LENGTH)
    holdout_ds = LeanDataset(holdout_texts, holdout_labels, tok, MAX_LENGTH)

    train_dl   = DataLoader(train_ds,   batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_dl     = DataLoader(val_ds,     batch_size=args.batch_size, shuffle=False, num_workers=0)
    holdout_dl = DataLoader(holdout_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = AutoModelForSequenceClassification.from_pretrained(
        ENCODER,
        num_labels=len(LEAN_CANON),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps  = args.epochs * math.ceil(len(train_dl))
    warmup_steps = int(total_steps * WARMUP_RATIO)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    print(f"\nTraining {args.epochs} epoch(s), {len(train_dl)} steps/epoch\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_dl, 1):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()

            total_loss += loss.item()
            if step % 200 == 0 or step == len(train_dl):
                print(f"  epoch {epoch}  step {step}/{len(train_dl)}  "
                      f"avg_loss={total_loss/step:.4f}")

        val_acc, val_mf1, _, _ = evaluate(model, val_dl)
        print(f"[epoch {epoch}] val acc={val_acc:.4f}  macro-F1={val_mf1:.4f}\n")

    # ── final evaluation on holdout ───────────────────────────────────────────
    print("="*60)
    print("HOLDOUT EVALUATION")
    print("="*60)

    _, _, preds, golds = evaluate(model, holdout_dl)
    pred_names = [ID2LABEL[p] for p in preds]
    gold_names = [ID2LABEL[g] for g in golds]

    acc = accuracy_score(golds, preds)
    mf1 = f1_score(golds, preds, average="macro", zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {mf1:.4f}")
    print()
    print(classification_report(gold_names, pred_names, labels=LEAN_CANON, zero_division=0))

    cm = confusion_matrix(gold_names, pred_names, labels=LEAN_CANON)
    print("Confusion matrix (rows=true, cols=pred):")
    header = "  " + "  ".join(f"{c[:6]:>7}" for c in LEAN_CANON)
    print(header)
    for i, row_vals in enumerate(cm):
        print(f"  {LEAN_CANON[i][:6]:>6}  " + "  ".join(f"{v:>7}" for v in row_vals))

    plot_cm(cm, LEAN_CANON,
            str(OUT_DIR / "cm_bert_baseline.png"),
            "BERT baseline — counts")
    plot_cm(cm, LEAN_CANON,
            str(OUT_DIR / "cm_bert_baseline_norm.png"),
            "BERT baseline — normalised", normalize=True)

    metrics = {
        "model":     ENCODER,
        "max_length": MAX_LENGTH,
        "epochs":    args.epochs,
        "lr":        args.lr,
        "accuracy":  float(acc),
        "macro_f1":  float(mf1),
        "confusion_matrix": cm.tolist(),
        "labels":    LEAN_CANON,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nAll outputs saved to {OUT_DIR}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=EPOCHS)
    ap.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    ap.add_argument("--lr",         type=float, default=LR)
    args = ap.parse_args()
    train(args)
