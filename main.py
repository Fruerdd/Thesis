import os
from pathlib import Path

# ---------------- PERFORMANCE / RESOURCE CONTROL ----------------
# Для скорости на твоём i7 можно поставить 8 (или 12), но оставляю умеренно:
CPU_THREADS = 8
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)

import torch
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)

# Если CUDA доступна — включаем ускорители для Ampere (RTX 3060 Ti)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# ---------------- CONFIG ----------------
MODEL_NAME = "bert-base-uncased"
DATA_DIR = Path("data")

FILE_ALLSIDES = DATA_DIR / "allsides.csv"
FILE_HEADLINES = DATA_DIR / "allsides_balanced_news_headlines-texts.csv"
FILE_ARTICLES = DATA_DIR / "newsmediabias-full.csv"

# ✅ СУПЕР ВАЖНО ДЛЯ СКОРОСТИ:
# Ограничиваем размер данных (train+val). С GPU можно поднять позже.
MAX_ARTICLE_SAMPLES = 200_000     # попробуй 50_000 если хочешь ещё быстрее
MAX_HEADLINE_SAMPLES = 200_000

VAL_SIZE = 0.05                  # 5% валид (быстрее)
MAX_LENGTH = 256                 # 128 заметно быстрее; 192/256 — медленнее

# Тренировать оба подряд — долго. На этапе отладки делай по одному.
TRAIN_ARTICLE_MODEL = True
TRAIN_HEADLINE_MODEL = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------- HELPERS ----------------
def detect_label_column(df: pd.DataFrame, candidates, dataset_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            print(f"[{dataset_name}] Using label column: '{c}'")
            return c
    raise ValueError(
        f"[{dataset_name}] None of {candidates} found. Available columns: {list(df.columns)}"
    )

def build_dataset_from_csv(
    csv_path: Path,
    text_columns,
    label_column: str,
    label_encoder: LabelEncoder | None = None,
    test_size: float = 0.2,
    max_samples: int | None = None,
):
    # Читаем только нужные колонки (ускоряет и экономит RAM)
    usecols_set = set(text_columns) | {label_column}

    df = pd.read_csv(
        csv_path,
        usecols=lambda c: c in usecols_set,
        low_memory=False,
    )

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {csv_path}. Got: {list(df.columns)}")

    available_text_cols = [c for c in text_columns if c in df.columns]
    if not available_text_cols:
        raise ValueError(
            f"No text columns from {text_columns} exist in {csv_path.name}. Got: {list(df.columns)}"
        )

    # Оставляем только строки с label и хотя бы одним непустым текстовым полем
    has_text = df[available_text_cols].notna().any(axis=1)
    df = df[df[label_column].notna() & has_text].reset_index(drop=True)

    # Encode labels (это быстро)
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df["label_id"] = label_encoder.fit_transform(df[label_column])
    else:
        df["label_id"] = label_encoder.transform(df[label_column])

    # ✅ КЛЮЧ: стратифицированный downsample ДО построения text_for_bert
    if max_samples is not None and len(df) > max_samples:
        df, _ = train_test_split(
            df,
            train_size=max_samples,
            random_state=42,
            stratify=df["label_id"],
        )
        df = df.reset_index(drop=True)
        print(f"[sampling] Using {len(df):,} rows from {csv_path.name}")

    # Быстро строим текст уже на маленьком df (без .apply по миллионам строк)
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

    df = df[df["text_for_bert"].str.len() > 0].reset_index(drop=True)

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
        padding=False,
        max_length=MAX_LENGTH,
    )

def prepare_for_trainer(train_ds: Dataset, val_ds: Dataset):
    cols_to_keep = ["input_ids", "attention_mask", "label_id"]

    # batched=True даёт ускорение токенизации :contentReference[oaicite:2]{index=2}
    # num_proc на Windows иногда может даже замедлять, поэтому оставляем 1. :contentReference[oaicite:3]{index=3}
    train_ds = train_ds.map(tokenize_batch, batched=True, batch_size=1024, num_proc=1)
    val_ds = val_ds.map(tokenize_batch, batched=True, batch_size=1024, num_proc=1)

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

# ---------------- DATASET 1: SOURCE → BIAS MAP ----------------
def build_source_bias_mapping():
    print("\n=== BUILDING SOURCE → BIAS MAP (DATASET 1) ===")
    df = pd.read_csv(FILE_ALLSIDES)
    print("[allsides] Columns:", list(df.columns))

    name_col = "name" if "name" in df.columns else "source"
    bias_col = "bias" if "bias" in df.columns else ("bias_rating" if "bias_rating" in df.columns else None)
    if bias_col is None:
        raise ValueError(f"[allsides] Could not find bias column. Got: {list(df.columns)}")

    mapping = dict(zip(df[name_col], df[bias_col]))
    print(f"Loaded {len(mapping)} sources from allsides.csv")
    return mapping

# ---------------- DATASET 3: ARTICLE-LEVEL ----------------
def train_article_bias_model():
    print("\n=== TRAINING ARTICLE-LEVEL BIAS MODEL (DATASET 3) ===")

    df_preview = pd.read_csv(FILE_ARTICLES, nrows=5)
    print("[articles] Columns:", list(df_preview.columns))

    candidate_labels = ["bias_rating", "bias", "label", "overall_bias"]
    label_col = detect_label_column(df_preview, candidate_labels, "articles")

    # В твоём файле реально есть "text" — не держим лишние столбцы
    text_cols = ["text"]

    train_ds, val_ds, le = build_dataset_from_csv(
        FILE_ARTICLES,
        text_columns=text_cols,
        label_column=label_col,
        test_size=VAL_SIZE,
        max_samples=MAX_ARTICLE_SAMPLES,
    )
    print("[articles] Classes:", list(le.classes_))
    num_labels = len(le.classes_)

    train_ds, val_ds = prepare_for_trainer(train_ds, val_ds)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)

    # fp16 ускоряет и экономит память на GPU :contentReference[oaicite:4]{index=4}
    use_fp16 = (device == "cuda")

    training_args = TrainingArguments(
        output_dir="./bert_article_bias",
        eval_strategy="epoch",          # можно "no" для максимальной скорости :contentReference[oaicite:5]{index=5}
        save_strategy="no",             # убираем чекпойнты (быстрее)
        learning_rate=2e-5,
        per_device_train_batch_size=16 if use_fp16 else 8,
        per_device_eval_batch_size=32 if use_fp16 else 16,
        num_train_epochs=1,             # для очень быстрого прогона; потом поставишь 2-3
        weight_decay=0.01,
        logging_steps=200,
        fp16=use_fp16,
        tf32=(device == "cuda"),        # TrainingArguments поддерживает tf32 в новых версиях :contentReference[oaicite:6]{index=6}
        dataloader_num_workers=2 if device == "cuda" else 0,
        dataloader_pin_memory=(device == "cuda"),
        report_to="none",
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
    trainer.save_model("./bert_article_bias/final_model")

    le_path = Path("./bert_article_bias/label_encoder_classes.txt")
    le_path.parent.mkdir(parents=True, exist_ok=True)
    with le_path.open("w", encoding="utf-8") as f:
        for cls in le.classes_:
            f.write(str(cls) + "\n")

    print("Saved article model + labels to ./bert_article_bias")

# ---------------- DATASET 2: HEADLINES ----------------
def train_sentence_bias_model():
    print("\n=== TRAINING SENTENCE/HEADLINE BIAS MODEL (DATASET 2) ===")

    df_preview = pd.read_csv(FILE_HEADLINES, nrows=5)
    print("[headlines] Columns:", list(df_preview.columns))

    candidate_labels = ["label", "bias", "class", "stance", "bias_rating"]
    label_col = detect_label_column(df_preview, candidate_labels, "headlines")

    text_cols = ["text"]

    train_ds, val_ds, le = build_dataset_from_csv(
        FILE_HEADLINES,
        text_columns=text_cols,
        label_column=label_col,
        test_size=VAL_SIZE,
        max_samples=MAX_HEADLINE_SAMPLES,
    )
    print("[headlines] Classes:", list(le.classes_))
    num_labels = len(le.classes_)

    train_ds, val_ds = prepare_for_trainer(train_ds, val_ds)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)

    use_fp16 = (device == "cuda")

    training_args = TrainingArguments(
        output_dir="./bert_sentence_bias",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=32 if use_fp16 else 16,
        per_device_eval_batch_size=64 if use_fp16 else 32,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=200,
        fp16=use_fp16,
        tf32=(device == "cuda"),
        dataloader_num_workers=2 if device == "cuda" else 0,
        dataloader_pin_memory=(device == "cuda"),
        report_to="none",
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
    trainer.save_model("./bert_sentence_bias/final_model")

    le_path = Path("./bert_sentence_bias/label_encoder_classes.txt")
    le_path.parent.mkdir(parents=True, exist_ok=True)
    with le_path.open("w", encoding="utf-8") as f:
        for cls in le.classes_:
            f.write(str(cls) + "\n")

    print("Saved sentence/headline model + labels to ./bert_sentence_bias")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    _ = build_source_bias_mapping()

    if TRAIN_ARTICLE_MODEL:
        train_article_bias_model()

    if TRAIN_HEADLINE_MODEL:
        train_sentence_bias_model()
