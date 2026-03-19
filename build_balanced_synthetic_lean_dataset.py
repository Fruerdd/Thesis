import re
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
INPUT_CSV = Path("data_prepared/combined_lean_dataset_v2.csv")

OUT_DIR = Path("data_prepared")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FULL = OUT_DIR / "combined_lean_dataset_balanced_15000.csv"
OUT_SYNTH = OUT_DIR / "combined_lean_dataset_synthetic_only_15000.csv"
OUT_STATS = OUT_DIR / "combined_lean_dataset_balanced_15000_stats.json"

# ============================================================
# CONFIG
# ============================================================
SEED = 42
TARGET_ROWS_PER_CLASS = 15000

LEAN_CANON = ["Right", "Right-center", "Center", "Left-center", "Left"]
LEAN_PRETTY = {
    "Right": "right",
    "Right-center": "lean right",
    "Center": "center",
    "Left-center": "lean left",
    "Left": "left",
}

MIN_SYNTH_WORDS = 180
MAX_SYNTH_WORDS = 420
MIN_SENTENCE_WORDS = 6
SIM_THRESHOLD_EXACT = 1.0

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
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def split_sentences(text: str) -> List[str]:
    text = clean_text_value(text)
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if word_count(s) < MIN_SENTENCE_WORDS:
            continue
        out.append(s)
    return out


def dedupe_sentences(sentences: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in sentences:
        key = normalize_text_for_compare(s)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s.strip())
    return out


def safe_title_from_text(text: str, max_words: int = 14) -> str:
    first_sent = split_sentences(text)
    if not first_sent:
        return ""
    words = re.findall(r"\b\w+\b", first_sent[0])
    title = " ".join(words[:max_words]).strip()
    return title[:120]


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


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


# ============================================================
# WORD TARGETS
# ============================================================
def compute_target_total_words(df: pd.DataFrame) -> int:
    per_class_rows = df.groupby("lean").size().to_dict()
    per_class_words = df.groupby("lean")["text"].apply(lambda s: int(sum(word_count(x) for x in s))).to_dict()

    avg_words = []
    for lean in LEAN_CANON:
        rows = int(per_class_rows.get(lean, 0))
        words = int(per_class_words.get(lean, 0))
        if rows > 0:
            avg_words.append(words / rows)

    if not avg_words:
        return 300 * TARGET_ROWS_PER_CLASS

    clipped = [min(max(x, 220.0), 380.0) for x in avg_words]
    target_avg = int(round(float(np.median(clipped))))
    target_total_words = max(
        int(max(per_class_words.values())),
        target_avg * TARGET_ROWS_PER_CLASS
    )
    return target_total_words


# ============================================================
# SYNTH GENERATOR
# ============================================================
class ClassPool:
    def __init__(self, lean: str, df: pd.DataFrame):
        self.lean = lean
        self.df = df.reset_index(drop=True).copy()

        self.df["text_clean"] = self.df["text"].apply(clean_text_value)
        self.df["text_norm"] = self.df["text_clean"].apply(normalize_text_for_compare)
        self.df["wc"] = self.df["text_clean"].apply(word_count)
        self.df["sentences"] = self.df["text_clean"].apply(split_sentences)

        self.df = self.df[self.df["sentences"].apply(len) >= 3].reset_index(drop=True)

        self.real_norm_set = set(self.df["text_norm"].tolist())
        self.generated_norm_set = set()

        self.rows = self.df.to_dict("records")

        if not self.rows:
            raise RuntimeError(f"No usable rows for class {lean}")

        self.long_rows = [r for r in self.rows if r["wc"] >= 220]
        self.medium_rows = [r for r in self.rows if 120 <= r["wc"] < 220]
        self.short_rows = [r for r in self.rows if r["wc"] < 120]

        if not self.long_rows:
            self.long_rows = self.rows[:]
        if not self.medium_rows:
            self.medium_rows = self.rows[:]
        if not self.short_rows:
            self.short_rows = self.rows[:]

    def _pick_parents(self) -> List[Dict[str, Any]]:
        if len(self.rows) == 1:
            return [self.rows[0], self.rows[0], self.rows[0]]

        candidates = []
        candidates.append(random.choice(self.long_rows))
        candidates.append(random.choice(self.medium_rows))
        candidates.append(random.choice(self.rows))

        if len(self.rows) >= 4 and random.random() < 0.50:
            candidates.append(random.choice(self.rows))

        return candidates

    def _build_text_from_parents(self, target_words: int) -> Tuple[str, List[Any]]:
        parents = self._pick_parents()
        used_parent_ids = []
        out_sentences = []

        per_parent_budget = max(3, target_words // max(len(parents), 1))

        for parent in parents:
            sents = parent["sentences"][:]
            if len(sents) < 2:
                continue

            if len(sents) >= 5:
                take_n = random.randint(2, min(5, len(sents)))
            else:
                take_n = min(len(sents), random.randint(2, len(sents)))

            start_max = max(0, len(sents) - take_n)
            start = random.randint(0, start_max) if start_max > 0 else 0

            chosen = sents[start:start + take_n]

            local = []
            local_wc = 0
            for sent in chosen:
                sw = word_count(sent)
                if sw < MIN_SENTENCE_WORDS:
                    continue
                if local_wc + sw > per_parent_budget and len(local) >= 2:
                    break
                local.append(sent)
                local_wc += sw

            if local:
                used_parent_ids.append(parent.get("row_id", parent.get("id", "")))
                out_sentences.extend(local)

        out_sentences = dedupe_sentences(out_sentences)

        current_words = sum(word_count(s) for s in out_sentences)

        guard = 0
        while current_words < target_words and guard < 20:
            extra_parent = random.choice(self.long_rows if random.random() < 0.7 else self.rows)
            extra_sents = extra_parent["sentences"][:]
            if not extra_sents:
                guard += 1
                continue

            start = random.randint(0, max(0, len(extra_sents) - 2))
            for sent in extra_sents[start:start + 2]:
                key = normalize_text_for_compare(sent)
                existing = {normalize_text_for_compare(x) for x in out_sentences}
                if key and key not in existing and word_count(sent) >= MIN_SENTENCE_WORDS:
                    out_sentences.append(sent)
                    current_words += word_count(sent)
                    used_parent_ids.append(extra_parent.get("row_id", extra_parent.get("id", "")))
                    if current_words >= target_words:
                        break
            guard += 1

        out_sentences = dedupe_sentences(out_sentences)
        text = " ".join(out_sentences).strip()
        return text, used_parent_ids

    def generate_one(self, target_words: int, max_tries: int = 40) -> Optional[Dict[str, Any]]:
        target_words = clamp_int(target_words, MIN_SYNTH_WORDS, MAX_SYNTH_WORDS)

        for _ in range(max_tries):
            text, parent_ids = self._build_text_from_parents(target_words)

            if not is_valid_text(text):
                continue

            wc = word_count(text)
            if wc < MIN_SYNTH_WORDS:
                continue

            norm = normalize_text_for_compare(text)
            if not norm:
                continue

            if norm in self.real_norm_set or norm in self.generated_norm_set:
                continue

            self.generated_norm_set.add(norm)

            title = safe_title_from_text(text)

            return {
                "row_id": None,
                "title": title,
                "link": "",
                "topic": "",
                "date": "",
                "source_name": f"synthetic_{self.lean}",
                "text": text,
                "lean": self.lean,
                "dataset_name": "synthetic_recombined",
                "word_count": wc,
                "is_synthetic": 1,
                "parent_ids": "|".join(str(x) for x in parent_ids if str(x) != ""),
            }

        return None


# ============================================================
# MAIN BUILD
# ============================================================
def build_balanced_dataset(df_real: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df = df_real.copy()

    df["text"] = df["text"].apply(clean_text_value)
    df = df[df["lean"].isin(LEAN_CANON)].copy()
    df = df[df["text"].apply(is_valid_text)].copy()

    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df))

    if "title" not in df.columns:
        df["title"] = ""
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

    target_total_words = compute_target_total_words(df)
    print(f"\nTarget rows per class: {TARGET_ROWS_PER_CLASS}")
    print(f"Target total words per class: {target_total_words}")

    synthetic_rows = []

    for lean in LEAN_CANON:
        part = df[df["lean"] == lean].copy()
        current_rows = int(len(part))
        current_words = int(part["word_count"].sum())

        need_rows = max(0, TARGET_ROWS_PER_CLASS - current_rows)
        need_words = max(0, target_total_words - current_words)

        print(f"\n[{lean}] current rows={current_rows}, current words={current_words}")
        print(f"[{lean}] need synthetic rows={need_rows}, need extra words={need_words}")

        if need_rows == 0 and need_words == 0:
            continue

        pool = ClassPool(lean, part)

        synth_for_class = []
        rows_left = need_rows
        words_left = need_words

        safety = 0
        while rows_left > 0 and safety < need_rows * 30 + 200:
            avg_needed = math.ceil(words_left / max(rows_left, 1)) if words_left > 0 else MIN_SYNTH_WORDS
            target_words = clamp_int(avg_needed, MIN_SYNTH_WORDS, MAX_SYNTH_WORDS)

            created = pool.generate_one(target_words=target_words)
            if created is None:
                safety += 1
                continue

            synth_for_class.append(created)
            rows_left -= 1
            words_left = max(0, words_left - int(created["word_count"]))
            safety += 1

        if rows_left > 0:
            raise RuntimeError(
                f"Could not generate enough synthetic rows for {lean}. Missing {rows_left} rows."
            )

        synthetic_rows.extend(synth_for_class)

        final_wc = sum(int(x["word_count"]) for x in synth_for_class)
        print(f"[{lean}] generated rows={len(synth_for_class)}, generated words={final_wc}")

    df_synth = pd.DataFrame(synthetic_rows)

    if len(df_synth) == 0:
        raise RuntimeError("No synthetic rows were generated.")

    df_full = pd.concat([df, df_synth], ignore_index=True)

    df_full = df_full.reset_index(drop=True)
    df_full["row_id"] = np.arange(len(df_full))

    df_synth = df_synth.reset_index(drop=True)
    df_synth["row_id"] = np.arange(len(df_synth))

    stats = {
        "target_rows_per_class": TARGET_ROWS_PER_CLASS,
        "target_total_words_per_class": int(target_total_words),
        "full_dataset": summarize(df_full),
        "synthetic_only_dataset": summarize(df_synth),
    }

    return df_full, df_synth, stats


# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(SEED)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_CSV}\n"
            f"Run your combined dataset builder first."
        )

    print(f"Loading input dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    print(f"Loaded rows: {len(df)}")
    if "lean" not in df.columns or "text" not in df.columns:
        raise ValueError("Input dataset must contain at least 'lean' and 'text' columns.")

    real_stats = summarize(
        df.assign(
            text=df["text"].apply(clean_text_value),
            is_synthetic=0
        )[df["text"].apply(is_valid_text)]
    )
    print_summary(real_stats, "REAL INPUT DATASET SUMMARY")

    df_full, df_synth, stats = build_balanced_dataset(df)

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