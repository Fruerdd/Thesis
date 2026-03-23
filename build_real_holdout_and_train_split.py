import re
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ============================================================
# PATHS
# ============================================================
INPUT_CSV = Path("data_prepared/combined_lean_dataset_balanced_15000.csv")

OUT_DIR = Path("data_prepared")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_HOLDOUT = OUT_DIR / "combined_lean_real_holdout_5class_100_each.csv"
OUT_TRAIN_REMAINING = OUT_DIR / "combined_lean_train_without_holdout.csv"
OUT_STATS = OUT_DIR / "combined_lean_real_holdout_5class_100_each_stats.json"

# ============================================================
# CONFIG
# ============================================================
TARGET_PER_CLASS = 100
SEED = 42

# If True:
#   train output will also contain only real rows
# If False:
#   train output will contain all remaining rows (real + synthetic),
#   except the real holdout rows removed for testing
TRAIN_REAL_ONLY = False

LABELS = ["Right", "Right-center", "Center", "Left-center", "Left"]

BAD_TEXT_VALUES = {
    "",
    "null",
    "none",
    "nan",
    "error fetching article",
    "<null>",
    "n/a",
}

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


def is_valid_text(text: str, min_len: int = 30, min_alpha: int = 15) -> bool:
    s = clean_text_value(text)
    if not s:
        return False
    if len(s) < min_len:
        return False
    alpha = sum(1 for c in s if c.isalpha())
    return alpha >= min_alpha


def normalize_label(x: Any) -> Optional[str]:
    if x is None:
        return None

    s = str(x).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()

    if s in {"right", "conservative"}:
        return "Right"
    if s in {"right center", "center right", "lean right", "leaning right"}:
        return "Right-center"
    if s in {"center", "centre", "neutral", "centrist"}:
        return "Center"
    if s in {"left center", "center left", "lean left", "leaning left"}:
        return "Left-center"
    if s in {"left", "liberal"}:
        return "Left"

    return None


def detect_col(df: pd.DataFrame, candidates) -> Optional[str]:
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in col_map:
            return col_map[key]
    return None


# ============================================================
# MAIN
# ============================================================
def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    print(f"Loaded rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    text_col = detect_col(df, ["text", "article_text", "content", "body"])
    label_col = detect_col(df, ["lean", "label", "bias", "bias_rating"])
    title_col = detect_col(df, ["title", "headline"])
    source_col = detect_col(df, ["source_name", "source", "site", "publisher"])
    url_col = detect_col(df, ["url", "link", "article_url"])

    if text_col is None:
        raise ValueError("Could not find text column.")
    if label_col is None:
        raise ValueError("Could not find label/lean column.")
    if "is_synthetic" not in df.columns:
        raise ValueError("Input file must contain 'is_synthetic' column.")

    work = df.copy().reset_index(drop=True)
    work["source_row_id"] = work.index

    work["text"] = work[text_col].apply(clean_text_value)
    work["label"] = work[label_col].apply(normalize_label)

    if title_col is not None:
        work["title"] = work[title_col].fillna("").astype(str)
    else:
        work["title"] = ""

    if source_col is not None:
        work["source_name"] = work[source_col].fillna("").astype(str)
    else:
        work["source_name"] = ""

    if url_col is not None:
        work["url"] = work[url_col].fillna("").astype(str)
    else:
        work["url"] = ""

    work["is_synthetic"] = work["is_synthetic"].fillna(0).astype(int)

    # keep only valid labeled rows
    work = work[
        work["text"].apply(is_valid_text) &
        work["label"].isin(LABELS)
    ].copy().reset_index(drop=True)

    work["word_count"] = work["text"].str.findall(r"\b\w+\b").str.len()

    print("\nUsable rows per class (all rows):")
    print(work["label"].value_counts(dropna=False))

    print("\nUsable rows per class split by is_synthetic:")
    print(pd.crosstab(work["label"], work["is_synthetic"]))

    # --------------------------------------------------------
    # REAL-ONLY HOLDOUT POOL
    # --------------------------------------------------------
    real_pool = work[work["is_synthetic"] == 0].copy().reset_index(drop=True)

    print("\nReal-only usable rows per class:")
    print(real_pool["label"].value_counts(dropna=False))

    not_enough_real = {
        lab: int((real_pool["label"] == lab).sum())
        for lab in LABELS
        if int((real_pool["label"] == lab).sum()) < TARGET_PER_CLASS
    }
    if not_enough_real:
        raise RuntimeError(
            "Not enough REAL rows for one or more classes:\n" +
            "\n".join(f"  {lab}: {cnt}" for lab, cnt in not_enough_real.items())
        )

    # --------------------------------------------------------
    # SAMPLE REAL HOLDOUT: 100 PER CLASS
    # --------------------------------------------------------
    holdout_parts = []
    for label in LABELS:
        part = real_pool[real_pool["label"] == label].sample(
            n=TARGET_PER_CLASS,
            random_state=SEED
        )
        holdout_parts.append(part)

    holdout = pd.concat(holdout_parts, ignore_index=True)
    holdout = holdout.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    holdout["row_id"] = range(len(holdout))

    holdout_source_ids = set(holdout["source_row_id"].tolist())

    # --------------------------------------------------------
    # TRAINING DATASET = ORIGINAL USABLE DATA MINUS HOLDOUT
    # --------------------------------------------------------
    train_remaining = work[~work["source_row_id"].isin(holdout_source_ids)].copy().reset_index(drop=True)

    if TRAIN_REAL_ONLY:
        train_remaining = train_remaining[train_remaining["is_synthetic"] == 0].copy().reset_index(drop=True)

    train_remaining["row_id"] = range(len(train_remaining))

    # --------------------------------------------------------
    # OUTPUT COLUMNS
    # --------------------------------------------------------
    final_cols = [
        "row_id",
        "title",
        "url",
        "source_name",
        "text",
        "label",
        "word_count",
        "is_synthetic",
        "source_row_id",
    ]

    holdout = holdout[final_cols]
    train_remaining = train_remaining[final_cols]

    holdout.to_csv(OUT_HOLDOUT, index=False, encoding="utf-8")
    train_remaining.to_csv(OUT_TRAIN_REMAINING, index=False, encoding="utf-8")

    stats = {
        "input_csv": str(INPUT_CSV),
        "target_per_class_holdout": TARGET_PER_CLASS,
        "train_real_only": TRAIN_REAL_ONLY,

        "holdout_total_rows": int(len(holdout)),
        "holdout_rows_per_class": {
            lab: int((holdout["label"] == lab).sum())
            for lab in LABELS
        },
        "holdout_synthetic_rows_per_class": {
            lab: int(holdout[holdout["label"] == lab]["is_synthetic"].sum())
            for lab in LABELS
        },

        "train_total_rows": int(len(train_remaining)),
        "train_rows_per_class": {
            lab: int((train_remaining["label"] == lab).sum())
            for lab in LABELS
        },
        "train_synthetic_rows_per_class": {
            lab: int(train_remaining[train_remaining["label"] == lab]["is_synthetic"].sum())
            for lab in LABELS
        },

        "output_holdout_csv": str(OUT_HOLDOUT),
        "output_train_csv": str(OUT_TRAIN_REMAINING),
    }

    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\nDone.")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()