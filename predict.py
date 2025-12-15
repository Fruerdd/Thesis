import os
from pathlib import Path

# --------- resource limits ----------
CPU_THREADS = 8
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)

import torch
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# ---------------- CONFIG ----------------
MODEL_NAME = "bert-base-uncased"
DATA_DIR = Path("data")

FILE_HEADLINES = DATA_DIR / "allsides_balanced_news_headlines-texts.csv"

# speed knobs
MAX_SAMPLES = 200_000      # train+val total
VAL_SIZE = 0.05
MAX_LENGTH = 256

OUT_DIR = Path("./bert_political_bias_5cls")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------- label normalization ----------------
# You want: right, right-center, center, left-center, left
CANON = ["Right", "Right-center", "Center", "Left-center", "Left"]

def normalize_political_label(x: str) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None

    s = str(x).strip().lower()

    # common variants (AllSides style)
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

    # if dataset has only 3-way labels, keep them or return None
    # (But you asked for 5-way, so we drop unknowns)
    return None

def detect_label_column(df: pd.DataFrame, candidates, dataset_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            print(f"[{dataset_name}] Using label column: '{c}'")
            return c
    raise ValueError(f"[{dataset_name}] None of {candidates} found. Available: {list(df.columns)}")

def build_dataset_from_csv(
    csv_path: Path,
    text_columns,
    label_column: str,
    test_size: float,
    max_samples: int,
):
    # read only needed columns
    usecols_set = set(text_columns) | {label_column}
    df = pd.read_csv(csv_path, usecols=lambda c: c in usecols_set, low_memory=False)

    available_text_cols = [c for c in text_columns if c in df.columns]
    if not available_text_cols:
        raise ValueError(f"No text columns found among {text_columns}. Got: {list(df.columns)}")

    # normalize labels to 5-way
    df["label_norm"] = df[label_column].map(normalize_political_label)

    # keep only rows with valid label + some text
    has_text = df[available_text_cols].notna().any(axis=1)
    df = df[has_text & df["label_norm"].notna()].reset_index(drop=True)

    # stratified downsample BEFORE building text
    if len(df) > max_samples:
        df, _ = train_test_split(
            df,
            train_size=max_samples,
            random_state=42,
            stratify=df["label_norm"],
        )
        df = df.reset_index(drop=True)
        print(f"[sampling] Using {len(df):,} rows from {csv_path.name}")

    # build text fast
    if len(available_text_cols) == 1:
        col = available_text_cols[0]
        df["text_for_bert"] = df[col].astype(str)
    else:
        df["text_for_bert"] = (
            df[available_text_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # encode labels
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label_norm"])

    print("[labels] classes:", list(le.classes_))
    # sanity check: you want 5 classes
    if len(le.classes_) < 5:
        print(
            "\nWARNING: Your dataset does NOT contain all 5 classes after normalization.\n"
            "You got these classes only:", list(le.classes_), "\n"
            "To train 5-way (Right/Right-center/Center/Left-center/Left) you need data that contains Lean Left/Lean Right labels.\n"
        )

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["label_id"],
    )

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
    return train_ds, val_ds, le

def tokenize_batch(batch):
    return tokenizer(
        batch["text_for_bert"],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )

def prepare_for_trainer(train_ds: Dataset, val_ds: Dataset):
    cols_to_keep = ["input_ids", "attention_mask", "label_id"]

    train_ds = train_ds.map(tokenize_batch, batched=True, batch_size=1024)
    val_ds = val_ds.map(tokenize_batch, batched=True, batch_size=1024)

    train_ds = train_ds.rename_column("label_id", "labels").remove_columns(
        [c for c in train_ds.column_names if c not in cols_to_keep]
    )
    val_ds = val_ds.rename_column("label_id", "labels").remove_columns(
        [c for c in val_ds.column_names if c not in cols_to_keep]
    )
    return train_ds, val_ds

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

def train_political_bias_model():
    print("\n=== TRAINING 5-CLASS POLITICAL BIAS MODEL ===")
    df_preview = pd.read_csv(FILE_HEADLINES, nrows=5)
    print("[headlines] Columns:", list(df_preview.columns))

    # common candidates
    label_col = detect_label_column(df_preview, ["bias", "label", "class", "stance", "bias_rating"], "headlines")

    # most datasets use "text" column
    text_cols = ["text", "title"]  # will use only those that exist

    train_ds, val_ds, le = build_dataset_from_csv(
        FILE_HEADLINES,
        text_columns=text_cols,
        label_column=label_col,
        test_size=VAL_SIZE,
        max_samples=MAX_SAMPLES,
    )

    train_ds, val_ds = prepare_for_trainer(train_ds, val_ds)

    num_labels = len(le.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)

    use_fp16 = (device == "cuda")

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=32 if use_fp16 else 8,
        per_device_eval_batch_size=64 if use_fp16 else 16,
        num_train_epochs=1,          # fast; raise to 2 later if you want
        weight_decay=0.01,
        logging_steps=200,
        fp16=use_fp16,
        dataloader_num_workers=2 if device == "cuda" else 0,
        dataloader_pin_memory=(device == "cuda"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT_DIR / "final_model"))

    # save label order
    (OUT_DIR / "label_encoder_classes.txt").write_text(
        "\n".join(list(le.classes_)) + "\n",
        encoding="utf-8",
    )

    print("Saved 5-class political bias model to:", OUT_DIR)

def load_model_for_inference():
    model_dir = OUT_DIR / "final_model"
    labels = (OUT_DIR / "label_encoder_classes.txt").read_text(encoding="utf-8").splitlines()
    labels = [x for x in labels if x.strip()]

    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    mdl.eval()
    return tok, mdl, labels

def predict_political_bias(text: str, threshold_non_center: float = 0.5):
    tok, mdl, labels = load_model_for_inference()

    enc = tok(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()

    pred_id = int(probs.argmax())
    pred_label = labels[pred_id]

    prob_map = {labels[i]: float(probs[i]) for i in range(len(labels))}

    # biased/not derived from Center probability
    p_center = prob_map.get("Center", 0.0)
    biased_score = 1.0 - p_center
    biased = biased_score >= threshold_non_center

    return {
        "political_bias": pred_label,
        "biased": biased,
        "biased_score": float(biased_score),
        "probs": prob_map,
    }

if __name__ == "__main__":
    train_political_bias_model()

    # Example:
    # print(predict_political_bias("Paste an article text here...", threshold_non_center=0.6))
