from __future__ import annotations

import time
import json
from typing import Optional, Dict, Any

from db_pg import make_session_factory, ModelConfig, ArticleAnalysis
from url_extract import fetch_html, extract_title_and_text

# import your inference function
from train_bias_v2 import predict  # <-- if your predict() is in train_bias_v2.py


def ensure_model_config(
    session,
    *,
    model_dir: str,
    encoder_name: str,
    max_length: int,
    max_chunks: int,
    lean_labels: list[str],
    intensity_labels: list[str],
    run_name: Optional[str] = None,
    notes: Optional[str] = None,
    train_summary: Optional[dict] = None,
) -> int:
    """
    Reuse existing row if same (model_dir + encoder_name) exists, else create.
    """
    existing = (
        session.query(ModelConfig)
        .filter(ModelConfig.model_dir == model_dir, ModelConfig.encoder_name == encoder_name)
        .order_by(ModelConfig.id.desc())
        .first()
    )
    if existing:
        return int(existing.id)

    mc = ModelConfig(
        model_dir=model_dir,
        encoder_name=encoder_name,
        max_length=max_length,
        max_chunks=max_chunks,
        lean_labels=lean_labels,
        intensity_labels=intensity_labels,
        run_name=run_name,
        notes=notes,
        train_summary=train_summary,
    )
    session.add(mc)
    session.commit()
    return int(mc.id)


def analyze_text_and_store(
    *,
    db_url: str,
    model_config_id: int,
    source_name: str,
    text: str,
    encoder_name: str,
    model_dir: str,
    threshold: float = 0.5,
) -> int:
    Session = make_session_factory(db_url)
    t0 = time.time()

    # run model
    result = predict(text, model_dir=model_dir, encoder_path=encoder_name, threshold_non_center=threshold)
    duration_ms = int((time.time() - t0) * 1000)

    with Session() as session:
        row = ArticleAnalysis(
            model_config_id=model_config_id,
            source_name=source_name,
            input_type="text",
            url=None,
            http_status=None,
            fetch_error=None,
            title=None,
            raw_text=None,
            used_text=text,
            prediction=result,
            political_bias=result.get("political_bias"),
            bias_intensity=result.get("bias_intensity"),
            biased_score=float(result.get("biased_score", 0.0)),
            duration_ms=duration_ms,
        )
        session.add(row)
        session.commit()
        return int(row.id)


def analyze_url_and_store(
    *,
    db_url: str,
    model_config_id: int,
    source_name: str,
    url: str,
    encoder_name: str,
    model_dir: str,
    threshold: float = 0.5,
) -> int:
    Session = make_session_factory(db_url)
    t0 = time.time()

    html, status, err = fetch_html(url)
    title = None
    raw_text = ""

    if html:
        title, raw_text = extract_title_and_text(html)

    used_text = raw_text if raw_text else ""
    if not used_text:
        # still store the attempt (with error)
        used_text = ""

    if used_text:
        result = predict(used_text, model_dir=model_dir, encoder_path=encoder_name, threshold_non_center=threshold)
    else:
        result = {
            "error": "No text extracted",
            "political_bias": None,
            "bias_intensity": None,
            "biased": None,
            "biased_score": None,
        }

    duration_ms = int((time.time() - t0) * 1000)

    with Session() as session:
        row = ArticleAnalysis(
            model_config_id=model_config_id,
            source_name=source_name,
            input_type="url",
            url=url,
            http_status=status,
            fetch_error=err,
            title=title,
            raw_text=raw_text if raw_text else None,
            used_text=used_text,  # can be empty string; column is NOT NULL
            prediction=result,
            political_bias=result.get("political_bias"),
            bias_intensity=result.get("bias_intensity"),
            biased_score=(float(result["biased_score"]) if result.get("biased_score") is not None else None),
            duration_ms=duration_ms,
        )
        session.add(row)
        session.commit()
        return int(row.id)