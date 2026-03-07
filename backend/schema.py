from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal, Any, Dict


class AnalyzeRequest(BaseModel):
    source: str = Field(..., min_length=1)
    link_or_text: str = Field(..., min_length=1)
    threshold: float = 0.5

class AnalyzeResponse(BaseModel):
    id: int
    analyzed_at: Any

    source_name: str
    input_type: Literal["text", "url"]
    url: Optional[str] = None
    http_status: Optional[int] = None
    fetch_error: Optional[str] = None
    title: Optional[str] = None

    political_bias: Optional[str] = None
    bias_intensity: Optional[str] = None
    biased_score: Optional[float] = None

    prediction: Dict[str, Any]
    duration_ms: Optional[int] = None