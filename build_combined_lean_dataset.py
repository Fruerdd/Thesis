"""
Merge four raw lean-labeled CSVs in data/ into a single combined training file.

Inputs (column schemas are now known and used directly):
  data/Political_Bias.csv                          [Title, Link, Text, Source, Bias]
  data/Political_Bias_Update.csv                   [Title, Link, Text, Source, Bias]
  data/allsides_balanced_news_headlines-texts.csv  [title, heading, source, text, bias_rating]
  data/bias_clean.csv                              [url, topic, date, title, site, bias, page_text]

Output:
  data_prepared/combined_lean_dataset_v2.csv
  data_prepared/combined_lean_dataset_v2_stats.json
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path("data")
OUT_DIR = Path("data_prepared")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "combined_lean_dataset_v2.csv"
OUT_STATS = OUT_DIR / "combined_lean_dataset_v2_stats.json"

SIMILARITY_THRESHOLD = 0.80
MIN_TEXT_LEN = 30
MIN_ALPHA_CHARS = 15

LEAN_CANON = ["Right", "Right-center", "Center", "Left-center", "Left"]

LEAN_MAP = {
    "right": "Right",
    "conservative": "Right",
    "lean right": "Right-center",
    "leaning-right": "Right-center",
    "right-center": "Right-center",
    "center-right": "Right-center",
    "center": "Center",
    "neutral": "Center",
    "centrist": "Center",
    "lean left": "Left-center",
    "leaning-left": "Left-center",
    "left-center": "Left-center",
    "center-left": "Left-center",
    "left": "Left",
    "liberal": "Left",
}

BAD_TEXT = {"", "null", "none", "nan", "n/a", "<null>", "error fetching article"}


def norm_lean(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return LEAN_MAP.get(str(x).strip().lower())


def clean_text(x):
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in BAD_TEXT else s


def is_valid_text(s):
    if not s or len(s) < MIN_TEXT_LEN:
        return False
    return sum(1 for c in s if c.isalpha()) >= MIN_ALPHA_CHARS


def normalize_text_for_dedupe(text):
    t = clean_text(text).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def word_count(text):
    return len(re.findall(r"\b\w+\b", str(text)))


# ============================================================
# LOADERS — one per raw file, using known column names directly
# ============================================================
def load_political_bias(path: Path, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return pd.DataFrame({
        "title": df["Title"].fillna("").astype(str),
        "link": df["Link"].fillna("").astype(str),
        "topic": "",
        "date": "",
        "source_name": df["Source"].fillna("").astype(str),
        "text": df["Text"].apply(clean_text),
        "lean": df["Bias"].map(norm_lean),
        "dataset_name": dataset_name,
    })


def load_allsides() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "allsides_balanced_news_headlines-texts.csv", low_memory=False)
    return pd.DataFrame({
        "title": df["title"].fillna("").astype(str),
        "link": "",
        "topic": "",
        "date": "",
        "source_name": df["source"].fillna("").astype(str),
        "text": df["text"].apply(clean_text),
        "lean": df["bias_rating"].map(norm_lean),
        "dataset_name": "allsides_balanced",
    })


def load_bias_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "bias_clean.csv", low_memory=False)
    return pd.DataFrame({
        "title": df["title"].fillna("").astype(str),
        "link": df["url"].fillna("").astype(str),
        "topic": df["topic"].fillna("").astype(str),
        "date": df["date"].fillna("").astype(str),
        "source_name": df["site"].fillna("").astype(str),
        "text": df["page_text"].apply(clean_text),
        "lean": df["bias"].map(norm_lean),
        "dataset_name": "bias_clean",
    })


# ============================================================
# DEDUPLICATION
# ============================================================
def exact_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_norm"] = df["text"].apply(normalize_text_for_dedupe)
    df = df[df["text_norm"] != ""]
    return df.drop_duplicates(subset=["text_norm"], keep="first").reset_index(drop=True)


def near_dedupe_with_tfidf(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Block by (first 8 normalized words + length bucket), then char-ngram
    TF-IDF cosine inside each block. Keep first row in each near-duplicate
    cluster.
    """
    df = df.copy()
    df["block_key"] = df["text_norm"].apply(
        lambda t: " ".join(t.split()[:8]) + f"__{len(t.split()) // 50}"
    )

    keep = []
    for _, block in df.groupby("block_key", sort=False):
        block = block.reset_index(drop=False)
        if len(block) == 1:
            keep.append(int(block.loc[0, "index"]))
            continue

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        X = vectorizer.fit_transform(block["text_norm"].tolist())

        dropped = set()
        for i in range(X.shape[0]):
            if i in dropped:
                continue
            keep.append(int(block.loc[i, "index"]))
            sims = (X[i] @ X[i + 1:].T).toarray().ravel()
            for rel_j, sim in enumerate(sims, start=i + 1):
                if sim >= threshold:
                    dropped.add(rel_j)

    return df.loc[sorted(set(keep))].reset_index(drop=True)


# ============================================================
# STATS
# ============================================================
def summarize(df: pd.DataFrame) -> dict:
    return {
        "total_rows": int(len(df)),
        "rows_per_lean": {
            lean: int((df["lean"] == lean).sum()) for lean in LEAN_CANON
        },
        "words_per_lean": {
            lean: int(df.loc[df["lean"] == lean, "text"].apply(word_count).sum())
            for lean in LEAN_CANON
        },
    }


# ============================================================
# MAIN
# ============================================================
def main():
    frames = [
        load_political_bias(DATA_DIR / "Political_Bias.csv", "political_bias"),
        load_political_bias(DATA_DIR / "Political_Bias_Update.csv", "political_bias_update"),
        load_allsides(),
        load_bias_clean(),
    ]

    df = pd.concat(frames, ignore_index=True)

    df = df[df["lean"].notna() & df["text"].apply(is_valid_text)].reset_index(drop=True)
    print(f"[after label+text filter] rows: {len(df)}")
    print(df["lean"].value_counts())

    df = exact_dedupe(df)
    print(f"[after exact dedupe] rows: {len(df)}")

    df = near_dedupe_with_tfidf(df, threshold=SIMILARITY_THRESHOLD)
    print(f"[after near-dedupe @ {SIMILARITY_THRESHOLD:.2f}] rows: {len(df)}")

    df["row_id"] = np.arange(len(df))
    df["word_count"] = df["text"].apply(word_count)

    df = df[[
        "row_id", "title", "link", "topic", "date", "source_name",
        "text", "lean", "dataset_name", "word_count",
    ]]

    stats = summarize(df)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nFinal: {stats['total_rows']} rows")
    for lean, n in stats["rows_per_lean"].items():
        print(f"  {lean}: {n}")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
