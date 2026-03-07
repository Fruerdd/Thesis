from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import BigInteger, Integer, Text, ForeignKey, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB


class Base(DeclarativeBase):
    pass


class ModelConfig(Base):
    __tablename__ = "model_configs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    model_dir: Mapped[str] = mapped_column(Text, nullable=False)
    encoder_name: Mapped[str] = mapped_column(Text, nullable=False)
    max_length: Mapped[int] = mapped_column(Integer, nullable=False)
    max_chunks: Mapped[int] = mapped_column(Integer, nullable=False)

    lean_labels: Mapped[dict] = mapped_column(JSONB, nullable=False)          # store list as JSON
    intensity_labels: Mapped[dict] = mapped_column(JSONB, nullable=False)     # store list as JSON

    run_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    train_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    analyses: Mapped[list["ArticleAnalysis"]] = relationship(back_populates="model_config")


class ArticleAnalysis(Base):
    __tablename__ = "article_analyses"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    analyzed_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # IMPORTANT: matches your DB schema (model_config_id), NOT model_run_id
    model_config_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("model_configs.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    source_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    input_type: Mapped[str] = mapped_column(Text, nullable=False)  # "text" | "url"

    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fetch_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    used_text: Mapped[str] = mapped_column(Text, nullable=False)

    prediction: Mapped[dict] = mapped_column(JSONB, nullable=False)

    political_bias: Mapped[str | None] = mapped_column(Text, nullable=True)
    bias_intensity: Mapped[str | None] = mapped_column(Text, nullable=True)
    biased_score: Mapped[float | None] = mapped_column(nullable=True)

    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    model_config: Mapped["ModelConfig"] = relationship(back_populates="analyses")