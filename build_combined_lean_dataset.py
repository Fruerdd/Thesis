import re
import json
from pathlib import Path
from typing import Any, List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# PATHS
# ============================================================
DATA_DIR = Path("data")

FILE_ALLSIDES_HEADLINES = DATA_DIR / "allsides_balanced_news_headlines-texts.csv"
FILE_POLITICAL_BIAS = DATA_DIR / "Political_Bias.csv"
FILE_POLITICAL_BIAS_UPDATE = DATA_DIR / "Political_Bias_Update.csv"
FILE_BIAS_CLEAN = DATA_DIR / "bias_clean.csv"

OUT_DIR = Path("data_prepared")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "combined_lean_dataset_v2.csv"
OUT_STATS_JSON = OUT_DIR / "combined_lean_dataset_v2_stats.json"

# ============================================================
# CONFIG
# ============================================================
SIMILARITY_THRESHOLD = 0.90
MIN_TEXT_LEN = 30
MIN_ALPHA_CHARS = 15

LEAN_CANON = ["Right", "Right-center", "Center", "Left-center", "Left"]

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
def detect_col(df: pd.DataFrame, candidates: List[str]) -> str:
    col_map = {str(col).strip().lower(): col for col in df.columns}
    for c in candidates:
        key = str(c).strip().lower()
        if key in col_map:
            return col_map[key]
    raise ValueError(f"None of {candidates} found. Available: {list(df.columns)}")


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


def clean_text_value(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in BAD_TEXT_VALUES:
        return ""
    return s


def is_valid_text(x: Any) -> bool:
    s = clean_text_value(x)
    if not s:
        return False
    if len(s) < MIN_TEXT_LEN:
        return False
    alpha = sum(1 for c in s if c.isalpha())
    return alpha >= MIN_ALPHA_CHARS


def normalize_text_for_dedupe(text: str) -> str:
    text = clean_text_value(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text).lower()))


# ============================================================
# DATA LOADERS
# ============================================================
def load_allsides() -> pd.DataFrame:
    if not FILE_ALLSIDES_HEADLINES.exists():
        raise FileNotFoundError(f"Missing file: {FILE_ALLSIDES_HEADLINES}")

    df = pd.read_csv(FILE_ALLSIDES_HEADLINES, low_memory=False)

    title_col = detect_col(df, ["title", "tittle", "heading"])
    text_col = detect_col(df, ["text", "heading", "title"])
    source_col = detect_col(df, ["source", "source_name", "name"])
    label_col = detect_col(df, ["bias_rating", "bias", "label"])

    df["text"] = df[text_col].apply(clean_text_value)
    df["lean"] = df[label_col].map(norm_lean)

    df = df[
        df["lean"].notna() &
        df["text"].apply(is_valid_text)
    ].copy()

    df["title"] = df[title_col].fillna("").astype(str)
    df["link"] = ""
    df["topic"] = ""
    df["date"] = ""
    df["source_name"] = df[source_col].fillna("").astype(str)
    df["dataset_name"] = "allsides_balanced"

    return df[[
        "title", "link", "topic", "date", "source_name",
        "text", "lean", "dataset_name"
    ]]


def load_political_bias_file(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, low_memory=False)

    title_col = detect_col(df, ["title"])
    link_col = detect_col(df, ["link"])
    text_col = detect_col(df, ["text"])
    source_col = detect_col(df, ["source"])
    label_col = detect_col(df, ["bias", "bias_rating"])

    df["text"] = df[text_col].apply(clean_text_value)
    df["lean"] = df[label_col].map(norm_lean)

    df = df[
        df["lean"].notna() &
        df["text"].apply(is_valid_text)
    ].copy()

    df["title"] = df[title_col].fillna("").astype(str)
    df["link"] = df[link_col].fillna("").astype(str)
    df["topic"] = ""
    df["date"] = ""
    df["source_name"] = df[source_col].fillna("").astype(str)
    df["dataset_name"] = dataset_name

    return df[[
        "title", "link", "topic", "date", "source_name",
        "text", "lean", "dataset_name"
    ]]


def load_bias_clean() -> pd.DataFrame:
    if not FILE_BIAS_CLEAN.exists():
        raise FileNotFoundError(f"Missing file: {FILE_BIAS_CLEAN}")

    df = pd.read_csv(FILE_BIAS_CLEAN, low_memory=False)

    url_col = detect_col(df, ["url"])
    topic_col = detect_col(df, ["topic"])
    date_col = detect_col(df, ["date"])
    title_col = detect_col(df, ["title", "tittle"])
    site_col = detect_col(df, ["site", "source", "source_name", "name"])
    bias_col = detect_col(df, ["bias", "bias_rating"])
    text_col = detect_col(df, ["page_text", "text"])

    df["text"] = df[text_col].apply(clean_text_value)
    df["lean"] = df[bias_col].map(norm_lean)

    df = df[
        df["lean"].notna() &
        df["text"].apply(is_valid_text)
    ].copy()

    df["title"] = df[title_col].fillna("").astype(str)
    df["link"] = df[url_col].fillna("").astype(str)
    df["topic"] = df[topic_col].fillna("").astype(str)
    df["date"] = df[date_col].fillna("").astype(str)
    df["source_name"] = df[site_col].fillna("").astype(str)
    df["dataset_name"] = "bias_clean"

    return df[[
        "title", "link", "topic", "date", "source_name",
        "text", "lean", "dataset_name"
    ]]


# ============================================================
# PRIORITY + DEDUPE
# ============================================================
def add_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rarer classes first during near-deduplication.
    Then keep longer texts first.
    """
    counts = df["lean"].value_counts().to_dict()
    df = df.copy()
    df["class_count"] = df["lean"].map(counts).astype(int)
    df["text_words"] = df["text"].apply(word_count)

    df = df.sort_values(
        by=["class_count", "text_words"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return df


def exact_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_norm"] = df["text"].apply(normalize_text_for_dedupe)
    df = df[df["text_norm"] != ""].copy()
    df = df.drop_duplicates(subset=["text_norm"], keep="first").reset_index(drop=True)
    return df


def build_block_key(text_norm: str) -> str:
    words = text_norm.split()
    first_words = " ".join(words[:8]) if words else ""
    length_bucket = len(words) // 50
    return f"{first_words}__{length_bucket}"


def near_dedupe_with_tfidf(df: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """
    Approximate near-duplicate removal:
    - block by first 8 normalized words + length bucket
    - inside each block, use char-ngram TF-IDF cosine similarity
    - greedily keep first row, drop later rows if sim >= threshold
    """
    df = df.copy()
    df["block_key"] = df["text_norm"].apply(build_block_key)

    keep_indices = []

    for _, block in df.groupby("block_key", sort=False):
        block = block.reset_index(drop=False)
        texts = block["text_norm"].tolist()

        if len(block) == 1:
            keep_indices.append(int(block.loc[0, "index"]))
            continue

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        X = vectorizer.fit_transform(texts)

        kept_local = []
        dropped_local = set()

        for i in range(X.shape[0]):
            if i in dropped_local:
                continue

            kept_local.append(i)

            sims = (X[i] @ X[i + 1:].T).toarray().ravel()
            for rel_j, sim in enumerate(sims, start=i + 1):
                if sim >= threshold:
                    dropped_local.add(rel_j)

        keep_indices.extend(block.loc[kept_local, "index"].tolist())

    out = df.loc[sorted(set(keep_indices))].copy().reset_index(drop=True)
    return out


# ============================================================
# STATS
# ============================================================
def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    rows_per_lean = {}
    words_per_lean = {}

    for lean in LEAN_CANON:
        part = df[df["lean"] == lean]
        rows_per_lean[lean] = int(len(part))
        words_per_lean[lean] = int(part["text"].apply(word_count).sum())

    return {
        "total_rows": int(len(df)),
        "rows_per_lean": rows_per_lean,
        "words_per_lean": words_per_lean,
    }


def print_summary(stats: Dict[str, Any]) -> None:
    print("\n=== FINAL DATASET SUMMARY ===")
    print(f"Total rows: {stats['total_rows']}")

    print("\nRows per political leaning:")
    for lean, cnt in stats["rows_per_lean"].items():
        print(f"  {lean}: {cnt}")

    print("\nTotal words per political leaning:")
    for lean, cnt in stats["words_per_lean"].items():
        print(f"  {lean}: {cnt}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading datasets...")

    df_a = load_allsides()
    df_b = load_political_bias_file(FILE_POLITICAL_BIAS, "political_bias")
    df_c = load_political_bias_file(FILE_POLITICAL_BIAS_UPDATE, "political_bias_update")
    df_d = load_bias_clean()

    print(f"[allsides_balanced] usable rows: {len(df_a)}")
    print(df_a["lean"].value_counts())

    print(f"\n[political_bias] usable rows: {len(df_b)}")
    print(df_b["lean"].value_counts())

    print(f"\n[political_bias_update] usable rows: {len(df_c)}")
    print(df_c["lean"].value_counts())

    print(f"\n[bias_clean] usable rows: {len(df_d)}")
    print(df_d["lean"].value_counts())

    df = pd.concat([df_a, df_b, df_c, df_d], ignore_index=True)
    print(f"\n[combined before dedupe] rows: {len(df)}")

    df = add_priority(df)

    # exact duplicates
    df = exact_dedupe(df)
    print(f"[after exact text dedupe] rows: {len(df)}")

    # near duplicates
    df = near_dedupe_with_tfidf(df, threshold=SIMILARITY_THRESHOLD)
    print(f"[after near-duplicate dedupe @ {SIMILARITY_THRESHOLD:.2f}] rows: {len(df)}")

    # final cleanup columns
    df["row_id"] = np.arange(len(df))
    df["word_count"] = df["text"].apply(word_count)

    final_cols = [
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
    ]
    df = df[final_cols].reset_index(drop=True)

    stats = summarize(df)
    print_summary(stats)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    OUT_STATS_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nSaved dataset to: {OUT_CSV}")
    print(f"Saved stats to: {OUT_STATS_JSON}")


if __name__ == "__main__":
    main()