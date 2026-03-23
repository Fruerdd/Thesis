import re
import json
from pathlib import Path
from typing import Any, Optional, Dict

import pandas as pd

# ============================================================
# PATHS
# ============================================================
INPUT_CSV = Path("data_prepared/combined_lean_dataset_balanced_15000.csv")

OUT_DIR = Path("data_prepared")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "combined_lean_5class_100_each.csv"
OUT_STATS = OUT_DIR / "combined_lean_5class_100_each_stats.json"

# ============================================================
# CONFIG
# ============================================================
TARGET_PER_CLASS = 100
SEED = 42

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

    work = pd.DataFrame()
    work["text"] = df[text_col].apply(clean_text_value)
    work["label"] = df[label_col].apply(normalize_label)

    if title_col is not None:
        work["title"] = df[title_col].fillna("").astype(str)
    else:
        work["title"] = ""

    if source_col is not None:
        work["source_name"] = df[source_col].fillna("").astype(str)
    else:
        work["source_name"] = ""

    if url_col is not None:
        work["url"] = df[url_col].fillna("").astype(str)
    else:
        work["url"] = ""

    if "is_synthetic" in df.columns:
        work["is_synthetic"] = df["is_synthetic"].fillna(0).astype(int)
    else:
        work["is_synthetic"] = 0

    work = work[
        work["text"].apply(is_valid_text) &
        work["label"].isin(LABELS)
    ].copy().reset_index(drop=True)

    work["word_count"] = work["text"].str.findall(r"\b\w+\b").str.len()

    print("\nUsable rows per class:")
    print(work["label"].value_counts(dropna=False))

    missing = {
        label: int((work["label"] == label).sum())
        for label in LABELS
    }

    not_enough = {lab: cnt for lab, cnt in missing.items() if cnt < TARGET_PER_CLASS}
    if not_enough:
        raise RuntimeError(
            "Not enough rows for one or more classes:\n" +
            "\n".join(f"  {lab}: {cnt}" for lab, cnt in not_enough.items())
        )

    sampled_parts = []
    for label in LABELS:
        part = work[work["label"] == label].sample(n=TARGET_PER_CLASS, random_state=SEED)
        sampled_parts.append(part)

    result = pd.concat(sampled_parts, ignore_index=True)
    result = result.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    result["row_id"] = range(len(result))

    final_cols = [
        "row_id",
        "title",
        "url",
        "source_name",
        "text",
        "label",
        "word_count",
        "is_synthetic",
    ]
    result = result[final_cols]

    result.to_csv(OUT_CSV, index=False, encoding="utf-8")

    stats = {
        "input_csv": str(INPUT_CSV),
        "target_per_class": TARGET_PER_CLASS,
        "total_rows": int(len(result)),
        "rows_per_class": {
            lab: int((result["label"] == lab).sum())
            for lab in LABELS
        },
        "synthetic_rows_per_class": {
            lab: int(result[result["label"] == lab]["is_synthetic"].sum())
            for lab in LABELS
        },
        "avg_words_per_class": {
            lab: float(result[result["label"] == lab]["word_count"].mean())
            for lab in LABELS
        },
        "unique_sources": int(result["source_name"].nunique()),
        "output_csv": str(OUT_CSV),
    }

    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\nDone.")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()