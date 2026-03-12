import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Import from your training file (same repo)
from train_bias_v2 import (
    HierMultiTaskBiasModel,
    Batch,
    encode_to_chunks,
    extract_features,
    LEAN_CANON,
    INT_CANON,
)

# -----------------------
# DEVICE
# -----------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

USE_AMP = DEVICE in {"cuda", "mps"}
AMP_DTYPE = torch.float16 if DEVICE == "cuda" else (torch.bfloat16 if DEVICE == "mps" else None)

# -----------------------
# JSON parsing helpers
# -----------------------
def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def extract_article_fields(obj: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Returns (source_name, url, title, text)
    Works with your example format: obj["thread"]["site_full"], obj["url"], obj["title"], obj["text"].
    """
    source = (
        safe_get(obj, ["thread", "site_full"])
        or safe_get(obj, ["thread", "site"])
        or safe_get(obj, ["thread", "site_section"])
        or obj.get("site_full")
        or obj.get("site")
        or "unknown"
    )

    url = obj.get("url") or safe_get(obj, ["thread", "url"]) or ""
    title = obj.get("title") or safe_get(obj, ["thread", "title"]) or ""
    text = obj.get("text") or safe_get(obj, ["thread", "text"]) or ""

    # Useful: include title to add signal
    full_text = (str(title).strip() + "\n\n" + str(text).strip()).strip()
    return str(source), str(url), str(title), full_text

def iter_json_files(root: Path):
    # Your structure has nested folders (folder/folder/*.json). rglob handles it.
    yield from root.rglob("*.json")

# -----------------------
# Model loader (load once)
# -----------------------
def load_model_once(model_dir: str, encoder_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = HierMultiTaskBiasModel(
        encoder_name=encoder_name,
        feat_dim=12,
        n_lean=len(LEAN_CANON),
        n_int=len(INT_CANON),
        n_domain=2,
    ).to(DEVICE)

    sd = torch.load(Path(model_dir) / "model.pt", map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return tok, model

@torch.no_grad()
def predict_batch(
    tok,
    model: HierMultiTaskBiasModel,
    texts: List[str],
    max_length: int,
    max_chunks: int,
    threshold_non_center: float,
) -> List[Dict[str, Any]]:
    B = len(texts)

    ids_t = torch.zeros((B, max_chunks, max_length), dtype=torch.long)
    att_t = torch.zeros((B, max_chunks, max_length), dtype=torch.long)
    cm_t  = torch.zeros((B, max_chunks), dtype=torch.long)
    ft_t  = torch.zeros((B, 12), dtype=torch.float32)

    for i, t in enumerate(texts):
        ids, att, cm = encode_to_chunks(tok, t, max_length, max_chunks)
        ids_t[i] = torch.tensor(ids, dtype=torch.long)
        att_t[i] = torch.tensor(att, dtype=torch.long)
        cm_t[i]  = torch.tensor(cm, dtype=torch.long)
        ft_t[i]  = torch.tensor(extract_features(t), dtype=torch.float32)

    batch = Batch(
        input_ids=ids_t,
        attention_mask=att_t,
        chunk_mask=cm_t,
        feats=ft_t,
        y_lean=torch.full((B,), -100, dtype=torch.long),
        y_int=torch.full((B,), -100, dtype=torch.long),
        domain=torch.zeros((B,), dtype=torch.long),
        lean_soft=torch.zeros((B, len(LEAN_CANON)), dtype=torch.float32),
        has_lean_soft=torch.zeros((B,), dtype=torch.long),
    )

    if USE_AMP and AMP_DTYPE is not None:
        with torch.autocast(device_type=DEVICE, enabled=True, dtype=AMP_DTYPE):
            out = model(batch, grl_lambda=0.0)
    else:
        out = model(batch, grl_lambda=0.0)

    probs_lean = torch.softmax(out["logits_lean"], dim=-1).float().cpu().numpy()
    probs_int  = torch.softmax(out["logits_int"], dim=-1).float().cpu().numpy()

    results: List[Dict[str, Any]] = []
    for i in range(B):
        pl = probs_lean[i]
        pi = probs_int[i]

        pred_lean = LEAN_CANON[int(np.argmax(pl))]
        pred_int  = INT_CANON[int(np.argmax(pi))]

        p_center = float(pl[LEAN_CANON.index("Center")])
        biased_score = 1.0 - p_center
        biased = biased_score >= threshold_non_center

        results.append({
            "political_bias": pred_lean,
            "bias_intensity": pred_int,
            "biased": bool(biased),
            "biased_score": float(biased_score),
            "lean_conf": float(np.max(pl)),
            "int_conf": float(np.max(pi)),
            "probs_lean_json": json.dumps({LEAN_CANON[j]: float(pl[j]) for j in range(len(LEAN_CANON))}),
            "probs_int_json": json.dumps({INT_CANON[j]: float(pi[j]) for j in range(len(INT_CANON))}),
        })

    return results

# -----------------------
# Dataset folder discovery
# -----------------------
def find_dataset_folders(datasets_root: Path) -> List[Path]:
    """
    You have:
      DataSets/
        Politics_negative_x/
          Politics_negative_x/
            *.json
    We'll return the top-level dataset folder (Politics_negative_x).
    """
    folders = []
    for p in datasets_root.iterdir():
        if p.is_dir():
            folders.append(p)
    return sorted(folders)

# -----------------------
# Run one folder
# -----------------------
def run_one_folder(
    folder: Path,
    tok,
    model,
    out_csv: Path,
    max_length: int,
    max_chunks: int,
    batch_size: int,
    threshold: float,
    limit: int,
):
    rows = []
    buffer_texts: List[str] = []
    buffer_meta: List[Tuple[str, str, str, str, int]] = []

    total = 0
    skipped = 0

    for jp in iter_json_files(folder):
        if limit and total >= limit:
            break

        try:
            obj = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue

        source, url, title, text = extract_article_fields(obj)
        total += 1

        if not text or len(text.strip()) < 30:
            skipped += 1
            rows.append({
                "dataset_folder": folder.name,
                "file": str(jp),
                "source": source,
                "url": url,
                "title": title,
                "text_len": 0 if not text else len(text),
                "error": "empty_or_too_short",
            })
            continue

        buffer_texts.append(text)
        buffer_meta.append((str(jp), source, url, title, len(text)))

        if len(buffer_texts) >= batch_size:
            preds = predict_batch(tok, model, buffer_texts, max_length, max_chunks, threshold)
            for (f, s, u, t, tl), pr in zip(buffer_meta, preds):
                rows.append({
                    "dataset_folder": folder.name,
                    "file": f,
                    "source": s,
                    "url": u,
                    "title": t,
                    "text_len": tl,
                    **pr,
                })
            buffer_texts = []
            buffer_meta = []

    # flush last batch
    if buffer_texts:
        preds = predict_batch(tok, model, buffer_texts, max_length, max_chunks, threshold)
        for (f, s, u, t, tl), pr in zip(buffer_meta, preds):
            rows.append({
                "dataset_folder": folder.name,
                "file": f,
                "source": s,
                "url": u,
                "title": t,
                "text_len": tl,
                **pr,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # summary
    ok = df[df.get("error").isna()] if "error" in df.columns else df
    print(f"\n=== {folder.name} ===")
    print(f"Total json files read: {total}")
    print(f"Skipped (empty/short): {skipped}")
    if len(ok):
        print("Lean distribution:")
        print(ok["political_bias"].value_counts().to_string())
        print("Intensity distribution:")
        print(ok["bias_intensity"].value_counts().to_string())
        print("Avg lean_conf:", float(ok["lean_conf"].mean()))
        print("Avg int_conf:", float(ok["int_conf"].mean()))
        print("Avg biased_score:", float(ok["biased_score"].mean()))
    else:
        print("No valid predictions.")
    return df

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_root", default="DataSets", help="Path to your DataSets folder")
    ap.add_argument("--model_dir", required=True, help="Trained model dir (model.pt + tokenizer files)")
    ap.add_argument("--encoder", default="bert-base-uncased")
    ap.add_argument("--out_dir", default="batch_outputs", help="Folder for CSV outputs")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_chunks", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="Optional: max files per dataset (0 = all)")
    args = ap.parse_args()

    datasets_root = Path(args.datasets_root)
    if not datasets_root.exists():
        raise RuntimeError(f"datasets_root not found: {datasets_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading model once...")
    tok, model = load_model_once(args.model_dir, args.encoder)
    print("Model loaded.")

    dataset_folders = find_dataset_folders(datasets_root)
    if not dataset_folders:
        raise RuntimeError(f"No dataset folders found in {datasets_root}")

    all_dfs = []
    for folder in dataset_folders:
        out_csv = out_dir / f"{folder.name}.csv"
        df = run_one_folder(
            folder=folder,
            tok=tok,
            model=model,
            out_csv=out_csv,
            max_length=args.max_length,
            max_chunks=args.max_chunks,
            batch_size=args.batch_size,
            threshold=args.threshold,
            limit=args.limit,
        )
        all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)
    merged_csv = out_dir / "ALL_DATASETS_MERGED.csv"
    merged.to_csv(merged_csv, index=False)

    print("\n✅ Saved per-folder CSVs to:", out_dir)
    print("✅ Saved merged CSV:", merged_csv)

if __name__ == "__main__":
    main()