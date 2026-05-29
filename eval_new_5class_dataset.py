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


def normalize_5class_label(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None

    s = str(x).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())

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

    from matplotlib.colors import LinearSegmentedColormap
    light_blues = LinearSegmentedColormap.from_list(
        "light_blues", plt.cm.Blues(np.linspace(0.05, 0.65, 256))
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=light_blues)
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
                color="black",
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

    preds = []

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
            preds.append(LEAN_CANON[pred_idx])

    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data_prepared/combined_lean_5class_100_each.csv")
    parser.add_argument("--encoder", type=str, default="bert-base-uncased")
    parser.add_argument("--mode", type=str, default="teacher", choices=["teacher", "student", "combined"])

    parser.add_argument("--teacher_dir", type=str, default="./bias_system_v3/teacher_lean_v3")
    parser.add_argument("--student_dir", type=str, default="./bias_system_v3/student_mt_softlean_v3_seed42")

    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--source_col", type=str, default="source_name")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_source_prior", action="store_true")

    parser.add_argument("--out_json", type=str, default="combined_5class_eval_metrics.json")
    parser.add_argument("--out_png", type=str, default="combined_5class_confusion_matrix.png")
    parser.add_argument("--out_png_norm", type=str, default="combined_5class_confusion_matrix_normalized.png")
    parser.add_argument("--out_preds", type=str, default="combined_5class_predictions.csv")

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
    work["gold_label"] = df[args.label_col].apply(normalize_5class_label)

    if args.source_col in df.columns:
        work["source_name"] = df[args.source_col].fillna("").astype(str)
    else:
        work["source_name"] = ""

    print("\nRaw label counts:")
    print(df[args.label_col].value_counts(dropna=False))

    work = work[
        work["text"].apply(is_valid_text) &
        work["gold_label"].isin(LEAN_CANON)
    ].copy().reset_index(drop=True)

    work["row_id"] = np.arange(len(work))

    print("\nNormalized 5-class label counts:")
    print(work["gold_label"].value_counts(dropna=False))
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

    pred_labels = predict_5class_dataset(
        df=work,
        model_dir=model_dir,
        encoder_path=args.encoder,
        batch_size=args.batch_size,
        use_source_prior=args.use_source_prior,
    )

    work["pred_label"] = pred_labels

    y_true = work["gold_label"].tolist()
    y_pred = work["pred_label"].tolist()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LEAN_CANON, average="macro")
    report = classification_report(
        y_true,
        y_pred,
        labels=LEAN_CANON,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=LEAN_CANON)

    print("\nConfusion matrix (counts):")
    print(cm)

    metrics = {
        "dataset": str(csv_path),
        "mode": args.mode,
        "model_dir": model_dir,
        "encoder": args.encoder,
        "num_rows": int(len(work)),
        "accuracy_5class": float(acc),
        "macro_f1_5class": float(macro_f1),
        "use_source_prior": bool(args.use_source_prior),
        "labels_order_5class": LEAN_CANON,
        "confusion_matrix_5class": cm.tolist(),
        "classification_report_5class": report,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    work.to_csv(args.out_preds, index=False, encoding="utf-8")

    plot_confusion_matrix(
        cm=cm,
        labels=LEAN_CANON,
        out_path=args.out_png,
        title=f"5-Class Confusion Matrix ({args.mode})",
        normalize=False,
    )

    plot_confusion_matrix(
        cm=cm,
        labels=LEAN_CANON,
        out_path=args.out_png_norm,
        title=f"5-Class Normalized Confusion Matrix ({args.mode})",
        normalize=True,
    )

    print("\n=== RESULTS ===")
    print(f"5-class Accuracy: {acc:.4f}")
    print(f"5-class Macro F1: {macro_f1:.4f}")
    print(f"Saved metrics to: {args.out_json}")
    print(f"Saved predictions to: {args.out_preds}")
    print(f"Saved confusion matrix to: {args.out_png}")
    print(f"Saved normalized confusion matrix to: {args.out_png_norm}")


if __name__ == "__main__":
    main()