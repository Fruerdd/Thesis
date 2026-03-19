import re
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
REAL_INPUT_CSV = Path("data_prepared/combined_lean_dataset_v2.csv")
SYNTH_INPUT_CSV = Path("data/synthetic_political_news_42144_rows.csv")

OUT_DIR = Path("data_prepared")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FULL = OUT_DIR / "combined_lean_dataset_balanced_15000.csv"
OUT_SYNTH = OUT_DIR / "combined_lean_dataset_synthetic_only_15000.csv"
OUT_STATS = OUT_DIR / "combined_lean_dataset_balanced_15000_stats.json"

# ============================================================
# CONFIG
# ============================================================
TARGET_ROWS_PER_CLASS = 15000

LEAN_CANON = ["Right", "Right-center", "Center", "Left-center", "Left"]
LEAN_PRETTY = {
    "Right": "right",
    "Right-center": "lean right",
    "Center": "center",
    "Left-center": "lean left",
    "Left": "left",
}

BAD_TEXT_VALUES = {
    "",
    "null",
    "none",
    "nan",
    "error fetching article",
    "<null>",
    "n/a",
}

STRICT_NO_TEXT_OVERLAP = False  # set True if you want hard failure on overlap


# ============================================================
# HELPERS
# ============================================================
def clean_text_value(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in BAD_TEXT_VALUES:
        return ""
    return s


def normalize_text_for_compare(text: str) -> str:
    text = clean_text_value(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_text(text: str, min_len: int = 30, min_alpha: int = 15) -> bool:
    s = clean_text_value(text)
    if not s:
        return False
    if len(s) < min_len:
        return False
    alpha = sum(1 for c in s if c.isalpha())
    return alpha >= min_alpha


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text)))


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    rows_per_lean = {}
    words_per_lean = {}
    synth_rows_per_lean = {}

    for lean in LEAN_CANON:
        part = df[df["lean"] == lean]
        rows_per_lean[lean] = int(len(part))
        words_per_lean[lean] = int(part["text"].apply(word_count).sum())
        synth_rows_per_lean[lean] = int(part["is_synthetic"].sum())

    return {
        "total_rows": int(len(df)),
        "rows_per_lean": rows_per_lean,
        "words_per_lean": words_per_lean,
        "synthetic_rows_per_lean": synth_rows_per_lean,
    }


def print_summary(stats: Dict[str, Any], title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"Total rows: {stats['total_rows']}")

    print("\nRows per political leaning:")
    for lean in LEAN_CANON:
        print(f"  {LEAN_PRETTY[lean]} ({lean}): {stats['rows_per_lean'][lean]}")

    print("\nTotal words per political leaning:")
    for lean in LEAN_CANON:
        print(f"  {LEAN_PRETTY[lean]} ({lean}): {stats['words_per_lean'][lean]}")

    print("\nSynthetic rows per political leaning:")
    for lean in LEAN_CANON:
        print(f"  {LEAN_PRETTY[lean]} ({lean}): {stats['synthetic_rows_per_lean'][lean]}")


def safe_title_from_text(text: str, max_words: int = 14) -> str:
    text = clean_text_value(text)
    if not text:
        return ""
    first_sent = re.split(r"(?<=[.!?])\s+", text.strip())
    first_sent = first_sent[0] if first_sent else ""
    words = re.findall(r"\b\w+\b", first_sent)
    title = " ".join(words[:max_words]).strip()
    return title[:120]


# ============================================================
# LOADERS
# ============================================================
def load_real_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Real dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    if "lean" not in df.columns or "text" not in df.columns:
        raise ValueError("Real dataset must contain at least 'lean' and 'text' columns.")

    df = df.copy()
    df["text"] = df["text"].apply(clean_text_value)
    df = df[df["lean"].isin(LEAN_CANON)].copy()
    df = df[df["text"].apply(is_valid_text)].copy()

    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df))

    if "title" not in df.columns:
        df["title"] = df["text"].apply(safe_title_from_text)
    else:
        df["title"] = df["title"].fillna("").astype(str)

    if "link" not in df.columns:
        df["link"] = ""
    if "topic" not in df.columns:
        df["topic"] = ""
    if "date" not in df.columns:
        df["date"] = ""
    if "source_name" not in df.columns:
        df["source_name"] = ""
    if "dataset_name" not in df.columns:
        df["dataset_name"] = "real_combined"

    df["word_count"] = df["text"].apply(word_count)
    df["is_synthetic"] = 0
    df["parent_ids"] = ""

    return df


def load_external_synthetic_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = {"article_text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Synthetic CSV must contain columns {required}, got: {list(df.columns)}"
        )

    df = df.rename(columns={"article_text": "text", "label": "lean"}).copy()
    df["text"] = df["text"].apply(clean_text_value)
    df = df[df["lean"].isin(LEAN_CANON)].copy()
    df = df[df["text"].apply(is_valid_text)].copy()

    df["row_id"] = np.arange(len(df))
    df["title"] = df["text"].apply(safe_title_from_text)
    df["link"] = ""
    df["topic"] = ""
    df["date"] = ""
    df["source_name"] = "external_synthetic"
    df["dataset_name"] = "synthetic_external_15000"
    df["word_count"] = df["text"].apply(word_count)
    df["is_synthetic"] = 1
    df["parent_ids"] = ""

    return df[
        [
            "row_id",
            "title",
            "link",
            "topic",
            "date",
            "source_name",
            "text",
            "lean",
            "dataset_name",
            "word_count",
            "is_synthetic",
            "parent_ids",
        ]
    ]


# ============================================================
# VALIDATION
# ============================================================
def validate_real_counts(df_real: pd.DataFrame) -> None:
    too_large = []
    for lean in LEAN_CANON:
        count = int((df_real["lean"] == lean).sum())
        if count > TARGET_ROWS_PER_CLASS:
            too_large.append((lean, count))

    if too_large:
        msg = ", ".join(f"{lean}={count}" for lean, count in too_large)
        raise RuntimeError(
            f"Some real classes already exceed target {TARGET_ROWS_PER_CLASS}: {msg}"
        )


def compute_needed_counts(df_real: pd.DataFrame) -> Dict[str, int]:
    needed = {}
    for lean in LEAN_CANON:
        current = int((df_real["lean"] == lean).sum())
        needed[lean] = max(0, TARGET_ROWS_PER_CLASS - current)
    return needed


def validate_synth_counts(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    need = compute_needed_counts(df_real)
    have = {lean: int((df_synth["lean"] == lean).sum()) for lean in LEAN_CANON}

    mismatches = []
    for lean in LEAN_CANON:
        if need[lean] != have[lean]:
            mismatches.append(f"{lean}: need={need[lean]}, have={have[lean]}")

    if mismatches:
        raise RuntimeError(
            "Synthetic file does not match the missing rows after real-data cleaning:\n  "
            + "\n  ".join(mismatches)
        )

    return {"needed_per_lean": need, "synthetic_per_lean": have}


def validate_duplicate_overlap(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> None:
    real_norm = df_real["text"].apply(normalize_text_for_compare)
    synth_norm = df_synth["text"].apply(normalize_text_for_compare)

    synth_dup_count = int(synth_norm.duplicated().sum())
    if synth_dup_count > 0:
        raise RuntimeError(f"Synthetic dataset contains {synth_dup_count} duplicate normalized texts.")

    overlap = set(real_norm.tolist()) & set(synth_norm.tolist())
    overlap_count = len(overlap)

    if overlap_count > 0:
        msg = f"Found {overlap_count} normalized text overlaps between real and synthetic datasets."
        if STRICT_NO_TEXT_OVERLAP:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}")


def validate_final_counts(df_full: pd.DataFrame) -> None:
    bad = []
    for lean in LEAN_CANON:
        count = int((df_full["lean"] == lean).sum())
        if count != TARGET_ROWS_PER_CLASS:
            bad.append(f"{lean}={count}")

    if bad:
        raise RuntimeError(
            f"Final dataset is not perfectly balanced at {TARGET_ROWS_PER_CLASS} per class: "
            + ", ".join(bad)
        )


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Loading real dataset: {REAL_INPUT_CSV}")
    df_real = load_real_dataset(REAL_INPUT_CSV)
    validate_real_counts(df_real)

    real_stats = summarize(df_real)
    print_summary(real_stats, "REAL INPUT DATASET SUMMARY")

    needed_counts = compute_needed_counts(df_real)
    print("\nMissing rows that must be filled by synthetic data:")
    for lean in LEAN_CANON:
        print(f"  {lean}: {needed_counts[lean]}")

    print(f"\nLoading synthetic dataset: {SYNTH_INPUT_CSV}")
    df_synth = load_external_synthetic_dataset(SYNTH_INPUT_CSV)

    count_info = validate_synth_counts(df_real, df_synth)
    validate_duplicate_overlap(df_real, df_synth)

    df_full = pd.concat([df_real, df_synth], ignore_index=True).reset_index(drop=True)
    df_full["row_id"] = np.arange(len(df_full))

    df_synth = df_synth.reset_index(drop=True)
    df_synth["row_id"] = np.arange(len(df_synth))

    validate_final_counts(df_full)

    full_cols = [
        "row_id",
        "title",
        "link",
        "topic",
        "date",
        "source_name",
        "text",
        "lean",
        "dataset_name",
        "word_count",
        "is_synthetic",
        "parent_ids",
    ]

    df_full = df_full[full_cols]
    df_synth = df_synth[full_cols]

    stats = {
        "target_rows_per_class": TARGET_ROWS_PER_CLASS,
        "real_input_dataset": summarize(df_real),
        "synthetic_only_dataset": summarize(df_synth),
        "full_dataset": summarize(df_full),
        **count_info,
    }

    df_full.to_csv(OUT_FULL, index=False, encoding="utf-8")
    df_synth.to_csv(OUT_SYNTH, index=False, encoding="utf-8")
    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print_summary(stats["full_dataset"], "FINAL FULL DATASET SUMMARY")
    print_summary(stats["synthetic_only_dataset"], "FINAL SYNTHETIC-ONLY DATASET SUMMARY")

    print(f"\nSaved full dataset to: {OUT_FULL}")
    print(f"Saved synthetic-only dataset to: {OUT_SYNTH}")
    print(f"Saved stats to: {OUT_STATS}")


if __name__ == "__main__":
    main()