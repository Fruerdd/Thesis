import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from transformers import AutoTokenizer

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

# Dataset label meaning (given by you)
DATASET_ID_TO_LABEL = {
    0: "Left",
    1: "Left-center",
    2: "Center",
    3: "Right-center",
    4: "Right",
}

# Build remap: dataset_id -> model_id
# model uses LEAN_CANON indices, dataset uses DATASET_ID_TO_LABEL
LABEL_TO_MODEL_ID = {lbl: i for i, lbl in enumerate(LEAN_CANON)}
DATASET_ID_TO_MODEL_ID = {k: LABEL_TO_MODEL_ID[v] for k, v in DATASET_ID_TO_LABEL.items()}


def load_model_once(model_dir: str, encoder_name: str):
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
def predict_lean_batch(tok, model, texts: List[str], max_length: int, max_chunks: int) -> np.ndarray:
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

    return out["logits_lean"].argmax(dim=-1).detach().cpu().numpy()


def per_class_tp_fp_fn_tn(cm: np.ndarray):
    total = cm.sum()
    out = []
    for i in range(cm.shape[0]):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        out.append((tp, fp, fn, tn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data_test/political_bias.csv")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--encoder", default="bert-base-uncased")
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--max_chunks", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "text" not in df.columns or "label" not in df.columns:
        raise RuntimeError(f"CSV must have columns text,label. Got: {list(df.columns)}")

    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    # keep valid labels only
    df = df[df["label"].isin(DATASET_ID_TO_MODEL_ID.keys())].reset_index(drop=True)
    df = df[df["text"].str.strip().str.len() >= 10].reset_index(drop=True)

    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit].copy()

    # REMAP y_true into MODEL label indices
    y_true_model = df["label"].map(DATASET_ID_TO_MODEL_ID).astype(int).to_numpy()
    texts = df["text"].tolist()

    print("Device:", DEVICE)
    print("Model labels order (LEAN_CANON indices):", LEAN_CANON)
    print("Dataset label meaning:", DATASET_ID_TO_LABEL)
    print("Remap dataset_id -> model_id:", DATASET_ID_TO_MODEL_ID)

    tok, model = load_model_once(args.model_dir, args.encoder)
    print("Model loaded:", args.model_dir)

    preds = []
    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i : i + args.batch_size]
        preds.append(predict_lean_batch(tok, model, batch_texts, args.max_length, args.max_chunks))
    y_pred_model = np.concatenate(preds, axis=0)

    acc = accuracy_score(y_true_model, y_pred_model)
    cm = confusion_matrix(y_true_model, y_pred_model, labels=list(range(len(LEAN_CANON))))

    print("\n=== RESULTS (in MODEL label space) ===")
    print(f"Rows evaluated: {len(df)}")
    print(f"Accuracy: {acc:.4f}")

    print("\nConfusion matrix (rows=true, cols=pred) using LEAN_CANON order:")
    print("Order:", LEAN_CANON)
    print(cm)

    stats = per_class_tp_fp_fn_tn(cm)
    rows = []
    for i, (tp, fp, fn, tn) in enumerate(stats):
        rows.append({"class": LEAN_CANON[i], "TP": tp, "FP": fp, "FN": fn, "TN": tn})
    print("\nPer-class TP/FP/FN/TN:")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nClassification report (LEAN_CANON):")
    print(classification_report(y_true_model, y_pred_model, target_names=LEAN_CANON, digits=4))


if __name__ == "__main__":
    main()