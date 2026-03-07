import numpy as np
import torch
from functools import lru_cache

# We reuse your existing model code (predict function) from train_bias_v2.py
# IMPORTANT: make sure your train_bias_v2.py predict() already does `.float()` before `.numpy()`.
from train_bias_v2 import predict as core_predict  # type: ignore


@lru_cache(maxsize=1)
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict_text(
    text: str,
    *,
    model_dir: str,
    encoder_name: str,
    threshold: float,
) -> dict:
    """
    Safe wrapper around your predict() to ensure JSON serializable outputs
    even when AMP / bf16 is used.
    """
    res = core_predict(
        text,
        model_dir=model_dir,
        encoder_path=encoder_name,
        threshold_non_center=threshold,
    )

    # Force-safe types (no numpy scalars)
    def to_py(x):
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        return x

    res["biased_score"] = to_py(res.get("biased_score"))
    res["biased"] = bool(res.get("biased"))

    # ensure dict floats
    res["probs_lean"] = {k: float(v) for k, v in res["probs_lean"].items()}
    res["probs_int"] = {k: float(v) for k, v in res["probs_int"].items()}
    res["chunk_attention"] = [float(v) for v in res.get("chunk_attention", [])]

    return res