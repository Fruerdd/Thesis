import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
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

DATA_DIR = Path("data")
FILE_LRLL_1 = DATA_DIR / "Political_Bias.csv"
FILE_LRLL_2 = DATA_DIR / "Political_Bias_Update.csv"
FILE_HEADLINES = DATA_DIR / "allsides_balanced_news_headlines-texts.csv"


def _detect_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found. Available: {list(df.columns)}")


def _norm_lean_5way(x: Any) -> Optional[str]:
    """
    Maps your new CSV 'Bias' values to the SAME 5-way labels used by training:
      right -> Right
      lean right -> Right-center
      center -> Center
      lean left -> Left-center
      left -> Left
    Anything else -> None (drops junk like '<unset>', 'Bias', empty)
    """
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


def load_new_bias_df_5way() -> pd.DataFrame:
    """
    Loads BOTH Political_Bias.csv + Political_Bias_Update.csv,
    keeps ONLY rows mapped into 5-way LEAN_CANON.
    """
    dfs = []
    for fp in [FILE_LRLL_1, FILE_LRLL_2]:
        if fp.exists():
            dfs.append(pd.read_csv(fp, low_memory=False))

    if not dfs:
        raise FileNotFoundError("Political_Bias.csv / Political_Bias_Update.csv not found in ./data")

    df = pd.concat(dfs, ignore_index=True)

    text_col = _detect_col(df, ["Text", "text"])
    title_col = _detect_col(df, ["Title", "title"])
    bias_col = _detect_col(df, ["Bias", "bias"])
    source_col = _detect_col(df, ["Source", "source"])

    df["lean_norm"] = df[bias_col].map(_norm_lean_5way)
    df = df[df["lean_norm"].notna()].copy()

    df["title"] = df[title_col].fillna("").astype(str)
    df["text"] = df[text_col].fillna("").astype(str)
    df["source"] = df[source_col].fillna("unknown").astype(str)

    df["full_text"] = (df["title"].str.strip() + "\n\n" + df["text"].str.strip()).str.strip()
    df = df[df["full_text"].str.len() >= 30].copy()

    lean_to_id = {k: i for i, k in enumerate(LEAN_CANON)}
    df["y_lean"] = df["lean_norm"].map(lean_to_id).astype(int)

    return df[["source", "full_text", "y_lean"]].reset_index(drop=True)


def cap_per_class(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """
    Prevent the new dataset from dominating:
    keep at most `cap` rows per class.
    """
    if cap <= 0:
        return df

    rng = np.random.RandomState(seed)
    parts = []
    for y, g in df.groupby("y_lean"):
        if len(g) > cap:
            parts.append(g.sample(n=cap, random_state=rng))
        else:
            parts.append(g)
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    return out


def load_replay_balanced_5way(replay_size: int, seed: int) -> pd.DataFrame:
    """
    Balanced replay from AllSides headlines:
    EXACT equal per 5 classes to preserve old decision boundaries.
    """
    if replay_size <= 0:
        return pd.DataFrame(columns=["source", "full_text", "y_lean"])

    if not FILE_HEADLINES.exists():
        raise FileNotFoundError(f"Replay file not found: {FILE_HEADLINES}")

    df = pd.read_csv(FILE_HEADLINES, low_memory=False)
    text_col = _detect_col(df, ["text", "heading", "title"])
    label_col = _detect_col(df, ["bias_rating", "bias", "label", "class", "stance"])

    df["lean_norm"] = df[label_col].map(_norm_lean_5way)
    df = df[df["lean_norm"].notna() & df[text_col].notna()].copy()

    lean_to_id = {k: i for i, k in enumerate(LEAN_CANON)}
    df["y_lean"] = df["lean_norm"].map(lean_to_id).astype(int)

    df["source"] = "allsides"
    df["full_text"] = df[text_col].astype(str)

    per_class = max(1, replay_size // len(LEAN_CANON))
    rng = np.random.RandomState(seed)

    parts = []
    for cls_name in LEAN_CANON:
        y = lean_to_id[cls_name]
        g = df[df["y_lean"] == y]
        if len(g) == 0:
            continue
        take = min(per_class, len(g))
        parts.append(g.sample(n=take, random_state=rng))

    out = pd.concat(parts, ignore_index=True)

    # top-up if needed
    if len(out) < replay_size:
        remaining = replay_size - len(out)
        out = pd.concat(
            [out, df.sample(n=min(remaining, len(df)), random_state=rng)],
            ignore_index=True
        )

    out = out.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    out = out.iloc[:replay_size].copy()

    return out[["source", "full_text", "y_lean"]].reset_index(drop=True)


class TextLeanDataset(TorchDataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        return {"text": str(r["full_text"]), "y_lean": int(r["y_lean"])}


class LeanCollator:
    def __init__(self, tok, max_length: int, max_chunks: int):
        self.tok = tok
        self.max_length = max_length
        self.max_chunks = max_chunks

    def __call__(self, examples: List[Dict[str, Any]]) -> Batch:
        B = len(examples)
        C = self.max_chunks
        L = self.max_length

        ids_t = torch.zeros((B, C, L), dtype=torch.long)
        att_t = torch.zeros((B, C, L), dtype=torch.long)
        cm_t = torch.zeros((B, C), dtype=torch.long)
        ft_t = torch.zeros((B, 12), dtype=torch.float32)
        y = torch.zeros((B,), dtype=torch.long)

        for i, ex in enumerate(examples):
            t = ex["text"]
            ids, att, cm = encode_to_chunks(self.tok, t, L, C)
            ids_t[i] = torch.tensor(ids, dtype=torch.long)
            att_t[i] = torch.tensor(att, dtype=torch.long)
            cm_t[i] = torch.tensor(cm, dtype=torch.long)
            ft_t[i] = torch.tensor(extract_features(t), dtype=torch.float32)
            y[i] = int(ex["y_lean"])

        return Batch(
            input_ids=ids_t,
            attention_mask=att_t,
            chunk_mask=cm_t,
            feats=ft_t,
            y_lean=y,
            y_int=torch.full((B,), -100, dtype=torch.long),
            domain=torch.zeros((B,), dtype=torch.long),
            lean_soft=torch.zeros((B, len(LEAN_CANON)), dtype=torch.float32),
            has_lean_soft=torch.zeros((B,), dtype=torch.long),
        )


def set_trainable(model: HierMultiTaskBiasModel, mode: str):
    """
    mode:
      - head_only: train only lean head (+ gate_lean + norm)
      - head_plus_last2: train lean head + last 2 encoder layers
      - full: train all
    """
    for p in model.parameters():
        p.requires_grad = False

    for p in model.head_lean.parameters():
        p.requires_grad = True
    for p in model.gate_lean.parameters():
        p.requires_grad = True
    for p in model.norm.parameters():
        p.requires_grad = True

    if mode == "head_plus_last2":
        enc = model.encoder
        if hasattr(enc, "encoder") and hasattr(enc.encoder, "layer"):
            layers = enc.encoder.layer
            for layer in layers[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True
        else:
            for p in model.encoder.parameters():
                p.requires_grad = True

    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    counts = train_df["y_lean"].value_counts().to_dict()
    w = []
    for i in range(len(LEAN_CANON)):
        c = counts.get(i, 1)
        w.append(1.0 / float(c))
    w = np.array(w, dtype=np.float32)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def ce_weighted(logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is None:
        return F.cross_entropy(logits, targets.to(logits.device))
    return F.cross_entropy(logits, targets.to(logits.device), weight=weights.to(logits.device))


@torch.no_grad()
def eval_lean(model: HierMultiTaskBiasModel, dl: DataLoader) -> Dict[str, float]:
    model.eval()
    losses = []
    ys, ps = [], []

    for batch in dl:
        if USE_AMP and AMP_DTYPE is not None:
            with torch.autocast(device_type=DEVICE, enabled=True, dtype=AMP_DTYPE):
                out = model(batch, grl_lambda=0.0)
                loss = F.cross_entropy(out["logits_lean"], batch.y_lean.to(out["logits_lean"].device))
        else:
            out = model(batch, grl_lambda=0.0)
            loss = F.cross_entropy(out["logits_lean"], batch.y_lean.to(out["logits_lean"].device))

        losses.append(float(loss.item()))
        pred = out["logits_lean"].argmax(dim=-1).detach().cpu().numpy().tolist()
        ys.extend(batch.y_lean.cpu().numpy().tolist())
        ps.extend(pred)

    acc = float(sum(int(a == b) for a, b in zip(ys, ps)) / max(1, len(ys)))

    # macro f1
    f1s = []
    for c in range(len(LEAN_CANON)):
        tp = sum((yt == c and yp == c) for yt, yp in zip(ys, ps))
        fp = sum((yt != c and yp == c) for yt, yp in zip(ys, ps))
        fn = sum((yt == c and yp != c) for yt, yp in zip(ys, ps))
        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(2 * prec * rec / (prec + rec + 1e-12))

    return {"loss": float(np.mean(losses)), "acc": acc, "f1_macro": float(sum(f1s) / len(f1s))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--encoder", default="bert-base-uncased")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--max_chunks", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--replay_size", type=int, default=20000)

    # ✅ NEW: cap how many samples per class we take from Political_Bias*.csv
    ap.add_argument("--lrll_cap_per_class", type=int, default=1500)

    ap.add_argument("--freeze_mode", choices=["head_only", "head_plus_last2", "full"], default="head_only")
    ap.add_argument("--use_class_weights", action="store_true")

    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("[New CSV] Loading 5-way labeled dataset...")
    new_df = load_new_bias_df_5way()
    print("[New CSV] Raw counts:")
    print(new_df["y_lean"].value_counts())

    new_df = cap_per_class(new_df, cap=args.lrll_cap_per_class, seed=args.seed)
    print(f"[New CSV] After cap_per_class={args.lrll_cap_per_class}:")
    print(new_df["y_lean"].value_counts())

    replay = load_replay_balanced_5way(args.replay_size, seed=args.seed)
    print("[Replay] Counts:")
    print(replay["y_lean"].value_counts())

    full_df = pd.concat([new_df, replay], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    n_val = max(1, int(len(full_df) * args.val_frac))
    val_df = full_df.iloc[:n_val].copy()
    train_df = full_df.iloc[n_val:].copy()

    print(f"[Split] train={len(train_df)} val={len(val_df)}")

    tok = AutoTokenizer.from_pretrained(model_dir)

    model = HierMultiTaskBiasModel(
        encoder_name=args.encoder,
        feat_dim=12,
        n_lean=len(LEAN_CANON),
        n_int=len(INT_CANON),
        n_domain=2,
    ).to(DEVICE)

    sd = torch.load(model_dir / "model.pt", map_location=DEVICE)
    model.load_state_dict(sd)

    set_trainable(model, args.freeze_mode)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    train_ds = TextLeanDataset(train_df)
    val_ds = TextLeanDataset(val_df)

    collator = LeanCollator(tok, args.max_length, args.max_chunks)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collator)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    weights = compute_class_weights(train_df) if args.use_class_weights else None
    if weights is not None:
        print("[Loss] Using class weights:", weights.tolist())

    print(f"[Finetune] freeze_mode={args.freeze_mode}, lr={args.lr}")
    step = 0

    from tqdm.auto import tqdm

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(train_dl, start=1), total=len(train_dl), dynamic_ncols=True, desc=f"epoch {epoch}/{args.epochs}")

        for i, batch in pbar:
            if USE_AMP and AMP_DTYPE is not None:
                with torch.autocast(device_type=DEVICE, enabled=True, dtype=AMP_DTYPE):
                    out = model(batch, grl_lambda=0.0)
                    loss = ce_weighted(out["logits_lean"], batch.y_lean, weights) / args.grad_accum
            else:
                out = model(batch, grl_lambda=0.0)
                loss = ce_weighted(out["logits_lean"], batch.y_lean, weights) / args.grad_accum

            loss.backward()

            if (i % args.grad_accum == 0) or (i == len(train_dl)):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            step += 1
            pbar.set_postfix(loss=float(loss.item() * args.grad_accum), step=step)

        metrics = eval_lean(model, val_dl)
        print(f"[Val][epoch {epoch}] {metrics}")

    torch.save(model.state_dict(), out_dir / "model.pt")
    tok.save_pretrained(out_dir)
    (out_dir / "finetune_meta.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print("✅ Saved to:", out_dir)


if __name__ == "__main__":
    main()