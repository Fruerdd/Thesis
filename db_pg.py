from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Dict

from sqlalchemy import (
    create_engine,
    Column,
    BigInteger,
    Integer,
    Text,
    DateTime,
    ForeignKey,
    Float,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class ModelConfig(Base):
    __tablename__ = "model_configs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    model_dir = Column(Text, nullable=False)
    encoder_name = Column(Text, nullable=False)

    max_length = Column(Integer, nullable=False)
    max_chunks = Column(Integer, nullable=False)
    lean_labels = Column(JSONB, nullable=False)
    intensity_labels = Column(JSONB, nullable=False)

    run_name = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    train_summary = Column(JSONB, nullable=True)

    analyses = relationship("ArticleAnalysis", back_populates="model_config")


class ArticleAnalysis(Base):
    __tablename__ = "article_analyses"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    analyzed_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    model_config_id = Column(BigInteger, ForeignKey("model_configs.id"), nullable=False)
    model_config = relationship("ModelConfig", back_populates="analyses")

    source_name = Column(Text, nullable=False)
    input_type = Column(Text, nullable=False)  # 'text' | 'url'
    url = Column(Text, nullable=True)

    http_status = Column(Integer, nullable=True)
    fetch_error = Column(Text, nullable=True)
    title = Column(Text, nullable=True)

    raw_text = Column(Text, nullable=True)
    used_text = Column(Text, nullable=False)

    prediction = Column(JSONB, nullable=False)

    political_bias = Column(Text, nullable=True)
    bias_intensity = Column(Text, nullable=True)
    biased_score = Column(Float, nullable=True)

    duration_ms = Column(Integer, nullable=True)


def make_session_factory(db_url: str):
    """
    db_url example:
      postgresql+psycopg2://postgres:postgres@localhost:5432/thesis
    """
    engine = create_engine(db_url, future=True, pool_pre_ping=True)
    # If you already ran SQL DDL manually, you can skip create_all,
    # but it's safe to keep if tables match.
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)