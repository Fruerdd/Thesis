from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from backend.models import Base, ModelConfig


def make_engine(db_url: str):
    # future=True => SQLAlchemy 2.0 style
    return create_engine(db_url, future=True, pool_pre_ping=True)


def make_session_factory(db_url: str):
    engine = make_engine(db_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_or_create_model_config(
    db: Session,
    *,
    model_dir: str,
    encoder_name: str,
    max_length: int,
    max_chunks: int,
    lean_labels: list[str],
    intensity_labels: list[str],
    run_name: str | None = None,
):
    cfg = (
        db.query(ModelConfig)
        .filter(
            ModelConfig.model_dir == model_dir,
            ModelConfig.encoder_name == encoder_name,
            ModelConfig.max_length == max_length,
            ModelConfig.max_chunks == max_chunks,
        )
        .first()
    )
    if cfg:
        return cfg

    cfg = ModelConfig(
        model_dir=model_dir,
        encoder_name=encoder_name,
        max_length=max_length,
        max_chunks=max_chunks,
        lean_labels=lean_labels,
        intensity_labels=intensity_labels,
        run_name=run_name,
    )
    db.add(cfg)
    db.commit()
    db.refresh(cfg)
    return cfg