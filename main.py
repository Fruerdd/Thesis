import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import torch
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# ---------------- CONFIG ----------------

MODEL_NAME = "bert-base-uncased"      # change to "roberta-base", etc. if you want
DATA_DIR = Path("data")

FILE_ALLSIDES = DATA_DIR / "allsides.csv"
FILE_HEADLINES = DATA_DIR / "allsides_balanced_news_headlines-texts.csv"
FILE_ARTICLES = DATA_DIR / "newsmediabias-full.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("Using device:", device)


# ---------------- HELPERS ----------------

def detect_label_column(df: pd.DataFrame, candidates, dataset_name: str) -> str:
    """
    Try to find the first column from 'candidates' that exists in df.
    If none found, raise a clear error listing available columns.
    """
    for c in candidates:
        if c in df.columns:
            print(f"[{dataset_name}] Using label column: '{c}'")
            return c
    raise ValueError(
        f"[{dataset_name}] None of the candidate label columns {candidates} "
        f"were found. Available columns: {list(df.columns)}"
    )


def build_dataset_from_csv(
    csv_path: Path,
    text_columns,
    label_column: str,
    label_encoder: LabelEncoder | None = None,
    test_size: float = 0.2,
):
    """
    Generic helper:
    - reads csv_path
    - concatenates text_columns into 'text_for_bert'
    - encodes label_column with LabelEncoder
    - splits into train/val
    - returns HF Datasets + LabelEncoder
    """
    df = pd.read_csv(csv_path)

    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    def combine_text(row):
        parts = []
        for col in text_columns:
            if col in row and pd.notna(row[col]):
                parts.append(str(row[col]))
        return " ".join(parts)

    df["text_for_bert"] = df.apply(combine_text, axis=1)

    # Drop rows where text or label is missing
    df = df.dropna(subset=["text_for_bert", label_column])

    # Encode labels
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df["label_id"] = label_encoder.fit_transform(df[label_column])
    else:
        df["label_id"] = label_encoder.transform(df[label_column])

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["label_id"],
    )

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    return train_ds, val_ds, label_encoder


def tokenize_batch(batch):
    return tokenizer(
        batch["text_for_bert"],
        truncation=True,
        padding=False,   # padding done by data_collator
        max_length=256,  # can increase to 512 if needed
    )


def prepare_for_trainer(train_ds: Dataset, val_ds: Dataset):
    cols_to_keep = ["input_ids", "attention_mask", "label_id"]

    train_ds = train_ds.map(tokenize_batch, batched=True)
    val_ds = val_ds.map(tokenize_batch, batched=True)

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


# ---------------- DATASET 3: ARTICLE-LEVEL BIAS ----------------

def train_article_bias_model():
    """
    Trains BERT on news articles (newsmediabias-full.csv).
    Uses text columns: title + heading + text.
    Automatically detects the label column among common candidates.
    """
    print("\n=== TRAINING ARTICLE-LEVEL BIAS MODEL (DATASET 3) ===")

    # Peek at the columns to auto-detect label
    df_preview = pd.read_csv(FILE_ARTICLES, nrows=5)
    print("[articles] Columns:", list(df_preview.columns))

    # adjust this list if your label column has a different name
    candidate_labels = ["bias_rating", "bias", "label", "overall_bias"]
    label_col = detect_label_column(df_preview, candidate_labels, "articles")

    text_cols = ["title", "heading", "text"]  # these can be extended with "tags", "source", etc.

    train_ds, val_ds, le = build_dataset_from_csv(
        FILE_ARTICLES,
        text_columns=text_cols,
        label_column=label_col,
    )

    print("[articles] Classes:", list(le.classes_))
    num_labels = len(le.classes_)

    train_ds, val_ds = prepare_for_trainer(train_ds, val_ds)

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    ).to(device)

    training_args = TrainingArguments(
        output_dir="./bert_article_bias",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./bert_article_bias/best_model")

    # Save label names
    le_path = Path("./bert_article_bias/label_encoder_classes.txt")
    le_path.parent.mkdir(parents=True, exist_ok=True)
    with le_path.open("w") as f:
        for cls in le.classes_:
            f.write(str(cls) + "\n")

    print("Saved article model + labels to ./bert_article_bias")


# ---------------- DATASET 2: SENTENCE / HEADLINE BIAS ----------------

def train_sentence_bias_model():
    """
    Trains BERT on sentence/headline level data
    (allsides_balanced_news_headlines-texts.csv).
    Uses 'text' column for input and auto-detects the label column.
    """
    print("\n=== TRAINING SENTENCE/HEADLINE BIAS MODEL (DATASET 2) ===")

    df_preview = pd.read_csv(FILE_HEADLINES, nrows=5)
    print("[headlines] Columns:", list(df_preview.columns))

    # Typical label names; adjust if needed
    candidate_labels = ["label", "bias", "class", "stance", "bias_rating"]
    label_col = detect_label_column(df_preview, candidate_labels, "headlines")

    text_cols = ["text"]  # change if your text column is named differently

    train_ds, val_ds, le = build_dataset_from_csv(
        FILE_HEADLINES,
        text_columns=text_cols,
        label_column=label_col,
    )

    print("[headlines] Classes:", list(le.classes_))
    num_labels = len(le.classes_)

    train_ds, val_ds = prepare_for_trainer(train_ds, val_ds)

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    ).to(device)

    training_args = TrainingArguments(
        output_dir="./bert_sentence_bias",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./bert_sentence_bias/best_model")

    le_path = Path("./bert_sentence_bias/label_encoder_classes.txt")
    le_path.parent.mkdir(parents=True, exist_ok=True)
    with le_path.open("w") as f:
        for cls in le.classes_:
            f.write(str(cls) + "\n")

    print("Saved sentence/headline model + labels to ./bert_sentence_bias")


# ---------------- DATASET 1: SOURCE → BIAS MAP ----------------

def build_source_bias_mapping():
    """
    Builds a dict: news-outlet-name -> bias (from allsides.csv).
    Not a model, but useful metadata for analysis.
    """
    print("\n=== BUILDING SOURCE → BIAS MAP (DATASET 1) ===")
    df = pd.read_csv(FILE_ALLSIDES)
    print("[allsides] Columns:", list(df.columns))

    # Try common column names
    name_col = "name" if "name" in df.columns else "source"
    bias_col = "bias" if "bias" in df.columns else "bias_rating" if "bias_rating" in df.columns else None

    if bias_col is None:
        raise ValueError(
            "[allsides] Could not find a bias column. "
            f"Available columns: {list(df.columns)}"
        )

    mapping = dict(zip(df[name_col], df[bias_col]))
    print(f"Loaded {len(mapping)} sources from allsides.csv")
    return mapping


# ---------------- MAIN ----------------

if __name__ == "__main__":
    # 1) source-level mapping (for analysis, not used directly in training here)
    source_bias_map = build_source_bias_mapping()

    # 2) article-level model (dataset 3)
    train_article_bias_model()

    # 3) sentence/headline-level model (dataset 2)
    train_sentence_bias_model()
