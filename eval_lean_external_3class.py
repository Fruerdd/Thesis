import json
import argparse
from pathlib import Path
from typing import Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from train_bias_v3 import (
    LEAN_CANON,
    DEVICE,
    USE_AMP,
    AMP_DTYPE,
    MAX_LENGTH,
    MAX_CHUNKS,
    load_model,
    PseudoTextDataset,
    HierTextCollator,
    apply_source_prior_if_ambiguous,
)

BAD_TEXT_VALUES = {
    "",
    "null",
    "none",
    "nan",
    "error fetching article",
    "<null>",
    "n/a",
}

THREE_CLASS_CANON = ["Right", "Center", "Left"]


def clean_text_value(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in BAD_TEXT_VALUES:
        return ""
    return s


def is_valid_text(x: Any, min_len: int = 30) -> bool:
    s = clean_text_value(x)
    if not s:
        return False
    if len(s) < min_len:
        return False
    alpha = sum(1 for c in s if c.isalpha())
    return alpha >= 15


def normalize_external_3class_label(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None

    s = str(x).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())

    if s in {"right", "conservative"}:
        return "Right"
    if s in {"center", "centre", "neutral", "centrist"}:
        return "Center"
    if s in {"left", "liberal"}:
        return "Left"

    return None


def collapse_5_to_3(label: str) -> str:
    if label in {"Right", "Right-center"}:
        return "Right"
    if label == "Center":
        return "Center"
    if label in {"Left-center", "Left"}:
        return "Left"
    raise ValueError(f"Unknown 5-class label: {label}")


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    out_path: str,
    title: str,
    normalize: bool = False,
) -> None:
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
    else:
        cm_plot = cm

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_plot, interpolation="nearest")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    threshold = cm_plot.max() / 2.0 if cm_plot.size else 0.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            value = cm_plot[i, j]
            text_value = f"{value:.2f}" if normalize else str(int(value))
            ax.text(
                j,
                i,
                text_value,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def predict_5class_dataset(
    df: pd.DataFrame,
    model_dir: str,
    encoder_path: str,
    batch_size: int,
    use_source_prior: bool,
) -> List[str]:
    tok, model = load_model(model_dir, encoder_path)
    model.eval()

    ds = PseudoTextDataset(df[["row_id", "text", "source_name"]].copy())
    collator = HierTextCollator(tok, max_length=MAX_LENGTH, max_chunks=MAX_CHUNKS)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collator,
    )

    preds_5 = []

    for item in dl:
        source_names = item["source_name"]
        batch = item["batch"]

        with torch.autocast(device_type=DEVICE, enabled=USE_AMP, dtype=AMP_DTYPE):
            out = model(batch, grl_lambda=0.0)
            probs = torch.softmax(out["logits_lean"], dim=-1).float().cpu().numpy()

        for i in range(len(probs)):
            p = probs[i]
            if use_source_prior:
                p = apply_source_prior_if_ambiguous(p, source_names[i])
            pred_idx = int(np.argmax(p))
            preds_5.append(LEAN_CANON[pred_idx])

    return preds_5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="bert-base-uncased")
    parser.add_argument("--mode", type=str, default="combined", choices=["teacher", "student", "combined"])

    parser.add_argument("--teacher_dir", type=str, default="./bias_system_v3/teacher_lean_v3")
    parser.add_argument("--student_dir", type=str, default="./bias_system_v3/student_mt_softlean_v3_seed42")

    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--source_col", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_source_prior", action="store_true")

    parser.add_argument("--out_json", type=str, default="external_eval_3class_metrics.json")
    parser.add_argument("--out_png", type=str, default="external_eval_3class_confusion_matrix.png")
    parser.add_argument("--out_png_norm", type=str, default="external_eval_3class_confusion_matrix_normalized.png")
    parser.add_argument("--out_preds", type=str, default="external_eval_3class_predictions.csv")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column '{args.text_col}'. Available: {list(df.columns)}")
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column '{args.label_col}'. Available: {list(df.columns)}")

    work = pd.DataFrame()
    work["text"] = df[args.text_col].apply(clean_text_value)
    work["gold_label_3"] = df[args.label_col].apply(normalize_external_3class_label)

    if args.source_col is not None and args.source_col in df.columns:
        work["source_name"] = df[args.source_col].fillna("").astype(str)
    else:
        work["source_name"] = ""

    print("\nRaw label counts:")
    print(df[args.label_col].value_counts(dropna=False))

    work = work[
        work["text"].apply(is_valid_text) &
        work["gold_label_3"].isin(THREE_CLASS_CANON)
    ].copy().reset_index(drop=True)

    work["row_id"] = np.arange(len(work))

    print("\nNormalized 3-class label counts:")
    print(work["gold_label_3"].value_counts(dropna=False))
    print(f"\nDevice: {DEVICE}")
    print(f"Usable rows: {len(work)}")

    if len(work) == 0:
        raise RuntimeError("No usable rows after cleaning and label normalization.")

    if args.mode == "teacher":
        model_dir = args.teacher_dir
    elif args.mode == "student":
        model_dir = args.student_dir
    else:
        model_dir = args.teacher_dir

    pred_labels_5 = predict_5class_dataset(
        df=work,
        model_dir=model_dir,
        encoder_path=args.encoder,
        batch_size=args.batch_size,
        use_source_prior=args.use_source_prior,
    )

    work["pred_label_5"] = pred_labels_5
    work["pred_label_3"] = work["pred_label_5"].apply(collapse_5_to_3)

    y_true = work["gold_label_3"].tolist()
    y_pred = work["pred_label_3"].tolist()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=THREE_CLASS_CANON, average="macro")
    report = classification_report(
        y_true,
        y_pred,
        labels=THREE_CLASS_CANON,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=THREE_CLASS_CANON)

    print("\nConfusion matrix (counts):")
    print(cm)

    metrics = {
        "dataset": str(csv_path),
        "mode": args.mode,
        "model_dir": model_dir,
        "encoder": args.encoder,
        "num_rows": int(len(work)),
        "accuracy_3class": float(acc),
        "macro_f1_3class": float(macro_f1),
        "use_source_prior": bool(args.use_source_prior),
        "labels_order_3class": THREE_CLASS_CANON,
        "confusion_matrix_3class": cm.tolist(),
        "classification_report_3class": report,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    work.to_csv(args.out_preds, index=False, encoding="utf-8")

    plot_confusion_matrix(
        cm=cm,
        labels=THREE_CLASS_CANON,
        out_path=args.out_png,
        title=f"3-Class Confusion Matrix ({args.mode})",
        normalize=False,
    )

    plot_confusion_matrix(
        cm=cm,
        labels=THREE_CLASS_CANON,
        out_path=args.out_png_norm,
        title=f"3-Class Normalized Confusion Matrix ({args.mode})",
        normalize=True,
    )

    print("\n=== RESULTS ===")
    print(f"3-class Accuracy: {acc:.4f}")
    print(f"3-class Macro F1: {macro_f1:.4f}")
    print(f"Saved metrics to: {args.out_json}")
    print(f"Saved predictions to: {args.out_preds}")
    print(f"Saved confusion matrix to: {args.out_png}")
    print(f"Saved normalized confusion matrix to: {args.out_png_norm}")


if __name__ == "__main__":
    main()