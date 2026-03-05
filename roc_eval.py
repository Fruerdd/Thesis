import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from train_bias_v2 import (
    DEVICE,
    BATCH_SIZE,
    MAX_LENGTH,
    MAX_CHUNKS,
    LEAN_CANON,
    INT_CANON,
    load_model,
    build_student_dataframe_only,
    StudentTextDataset,
    StudentHierTextCollator,
)


@torch.no_grad()
def collect_outputs(model, dl):
    model.eval()

    all_lean_y = []
    all_lean_prob = []

    all_int_y = []
    all_int_prob = []

    for batch in dl:
        out = model(batch, grl_lambda=0.0)

        probs_lean = torch.softmax(out["logits_lean"], dim=-1).float().cpu().numpy()
        probs_int = torch.softmax(out["logits_int"], dim=-1).float().cpu().numpy()

        y_lean = batch.y_lean.cpu().numpy()
        y_int = batch.y_int.cpu().numpy()

        mask_lean = y_lean != -100
        if mask_lean.any():
            all_lean_y.extend(y_lean[mask_lean].tolist())
            all_lean_prob.extend(probs_lean[mask_lean].tolist())

        mask_int = y_int != -100
        if mask_int.any():
            all_int_y.extend(y_int[mask_int].tolist())
            all_int_prob.extend(probs_int[mask_int].tolist())

    return {
        "lean_y": np.array(all_lean_y, dtype=np.int64),
        "lean_prob": np.array(all_lean_prob, dtype=np.float32),
        "int_y": np.array(all_int_y, dtype=np.int64),
        "int_prob": np.array(all_int_prob, dtype=np.float32),
    }


def plot_binary_bias_roc(lean_y, lean_prob, out_path: Path):
    center_idx = LEAN_CANON.index("Center")

    y_true = (lean_y != center_idx).astype(int)
    y_score = 1.0 - lean_prob[:, center_idx]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Binary ROC: Center vs Non-center")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    return {"binary_bias_auc": float(roc_auc)}


def plot_multiclass_roc(y_true, y_prob, class_names, title, out_path: Path):
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(8, 6))
    aucs = {}

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[class_names[i]] = float(roc_auc)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    return aucs


def build_loader(df, tokenizer):
    ds = StudentTextDataset(df)
    collator = StudentHierTextCollator(tokenizer, MAX_LENGTH, MAX_CHUNKS)

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collator,
    )


def evaluate_split(model, tokenizer, df, split_name: str, out_dir: Path):
    dl = build_loader(df, tokenizer)
    data = collect_outputs(model, dl)

    metrics = {}

    if len(data["lean_y"]) > 0:
        metrics.update(
            plot_binary_bias_roc(
                data["lean_y"],
                data["lean_prob"],
                out_dir / f"{split_name}_binary_bias_roc.png",
            )
        )

        metrics["lean_ovr_auc"] = plot_multiclass_roc(
            data["lean_y"],
            data["lean_prob"],
            LEAN_CANON,
            f"Lean One-vs-Rest ROC ({split_name})",
            out_dir / f"{split_name}_lean_ovr_roc.png",
        )

    if len(data["int_y"]) > 0:
        metrics["intensity_ovr_auc"] = plot_multiclass_roc(
            data["int_y"],
            data["int_prob"],
            INT_CANON,
            f"Intensity One-vs-Rest ROC ({split_name})",
            out_dir / f"{split_name}_intensity_ovr_roc.png",
        )

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--encoder", type=str, default="bert-base-uncased")
    ap.add_argument("--split", type=str, default="both", choices=["train", "val", "both"])
    ap.add_argument("--out_dir", type=str, default="roc_outputs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    tokenizer, model = load_model(args.model_dir, args.encoder)

    print("[ROC] rebuilding student train/val dataframes...")
    train_df, val_df = build_student_dataframe_only()

    summary = {}

    if args.split in {"train", "both"}:
        print("[ROC] evaluating TRAIN split...")
        summary["train"] = evaluate_split(model, tokenizer, train_df, "train", out_dir)

    if args.split in {"val", "both"}:
        print("[ROC] evaluating VAL split...")
        summary["val"] = evaluate_split(model, tokenizer, val_df, "val", out_dir)

    summary_path = out_dir / "roc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved ROC outputs to:", out_dir)
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()