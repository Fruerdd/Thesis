import time
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.settings import Settings
from backend.db import make_session_factory, get_or_create_model_config
from backend.models import ArticleAnalysis
from backend.schema import AnalyzeRequest, AnalyzeResponse
from backend.extract import extract_from_url
from backend.infer import predict_text
from typing import Optional
from fastapi import Query

LEAN_CANON = ["Right", "Right-center", "Center", "Left-center", "Left"]
INT_CANON  = ["Highly Biased", "Neutral", "Slightly Biased"]

# ✅ load from env
settings = Settings.from_env()
SessionLocal = make_session_factory(settings.db_url)

app = FastAPI(title="Bias Detector API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, db: Session = Depends(get_db)):

    t0 = time.time()
    payload = (req.link_or_text or "").strip()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty link_or_text")

    lower = payload.lower()
    input_type = "url" if lower.startswith("http://") or lower.startswith("https://") else "text"
    url = payload if input_type == "url" else None

    http_status = None
    fetch_error = None
    title = None
    raw_text = None
    used_text = None

    if input_type == "url":
        extracted = extract_from_url(url)
        http_status = extracted.http_status
        fetch_error = extracted.fetch_error
        title = extracted.title
        raw_text = extracted.raw_text
        used_text = extracted.used_text
    else:
        raw_text = payload
        used_text = payload

    if used_text is None:
        used_text = ""

    model_config = get_or_create_model_config(
        db,
        model_dir=settings.model_dir,
        encoder_name=settings.encoder_name,
        max_length=settings.max_length,
        max_chunks=settings.max_chunks,
        lean_labels=LEAN_CANON,
        intensity_labels=INT_CANON,
        run_name=settings.run_name,
    )

    result = None
    if used_text.strip():
        result = predict_text(
            used_text,
            model_dir=settings.model_dir,
            encoder_name=settings.encoder_name,
            threshold=req.threshold,
            source_name=req.source,
        )

    duration_ms = int((time.time() - t0) * 1000)

    prediction_dict = result or {
        "error": fetch_error or "No text extracted",
        "political_bias": None,
        "bias_intensity": None,
        "biased": None,
        "biased_score": None,
        "probs_lean": None,
        "probs_int": None,
        "chunk_attention": None,
    }

    political_bias = result.get("political_bias") if result else None
    bias_intensity = result.get("bias_intensity") if result else None
    biased_score = result.get("biased_score") if result else None

    row = ArticleAnalysis(
        model_config_id=model_config.id,
        source_name=req.source,
        input_type=input_type,
        url=url,
        http_status=http_status,
        fetch_error=fetch_error,
        title=title,
        raw_text=raw_text,
        used_text=used_text,
        prediction=prediction_dict,
        political_bias=political_bias,
        bias_intensity=bias_intensity,
        biased_score=biased_score,
        duration_ms=duration_ms,
    )

    db.add(row)
    db.commit()
    db.refresh(row)

    return AnalyzeResponse(
        id=row.id,
        analyzed_at=row.analyzed_at,
        source_name=row.source_name,
        input_type=row.input_type,
        url=row.url,
        http_status=row.http_status,
        fetch_error=row.fetch_error,
        title=row.title,
        political_bias=row.political_bias,
        bias_intensity=row.bias_intensity,
        biased_score=row.biased_score,
        prediction=row.prediction,
        duration_ms=row.duration_ms,
    )

@app.get("/analyses")
def list_analyses(
    domain: Optional[str] = Query(None),       # "headline" | "article"
    source: Optional[str] = Query(None),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(ArticleAnalysis).filter(
        ArticleAnalysis.political_bias.isnot(None),
        ArticleAnalysis.bias_intensity.isnot(None),
    )
    if domain:
        q = q.filter(ArticleAnalysis.input_type == domain)
    if source:
        q = q.filter(ArticleAnalysis.source_name == source)
    if min_confidence > 0:
        q = q.filter(ArticleAnalysis.biased_score >= min_confidence)

    q = q.order_by(ArticleAnalysis.analyzed_at.desc())
    if limit is not None:
        q = q.limit(limit)
    rows = q.all()

    result = []
    for r in rows:
        pred = r.prediction or {}
        probs_lean = pred.get("probs_lean") or {}
        probs_int  = pred.get("probs_int") or {}
        lean       = r.political_bias or "Center"
        intensity  = r.bias_intensity or "Neutral"

        used = r.used_text or ""
        word_count = len(used.split())
        if r.input_type == "url":
            domain = "article"
        elif word_count >= 30:
            domain = "article"
        else:
            domain = "headline"

        result.append({
            "id":                  r.id,
            "text":                (r.title or used or "")[:200],
            "domain":              domain,
            "source":              r.source_name,
            "sourceType":          "external_synthetic" if r.source_name == "external_synthetic" else "real",
            "lean":                lean,
            "leanConfidence":      float(probs_lean.get(lean, 0.0)),
            "intensity":           intensity,
            "intensityConfidence": float(probs_int.get(intensity, 0.0)),
            "confidence":          float(r.biased_score or 0.0),
            "chunkAttention":      pred.get("chunk_attention") or [],
        })

    return result


@app.get("/sources")
def list_sources(db: Session = Depends(get_db)):
    rows = (
        db.query(ArticleAnalysis.source_name)
        .filter(ArticleAnalysis.political_bias.isnot(None))
        .distinct()
        .all()
    )
    return sorted(r.source_name for r in rows)