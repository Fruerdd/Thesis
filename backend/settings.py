import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_url: str
    model_dir: str
    encoder_name: str
    max_length: int
    max_chunks: int
    run_name: str | None

    @staticmethod
    def from_env() -> "Settings":
        db_url = os.environ.get("BIAS_DB_URL") or os.environ.get("BIAS_DATABASE_URL")
        if not db_url:
            raise RuntimeError("BIAS_DB_URL is not set")

        model_dir = os.environ.get("BIAS_MODEL_DIR")
        if not model_dir:
            raise RuntimeError("BIAS_MODEL_DIR is not set")
        if not Path(model_dir).exists():
            raise RuntimeError(f"BIAS_MODEL_DIR does not exist: {model_dir}")

        encoder = os.environ.get("BIAS_ENCODER", "bert-base-uncased")
        max_length = int(os.environ.get("BIAS_MAX_LENGTH", "128"))
        max_chunks = int(os.environ.get("BIAS_MAX_CHUNKS", "2"))
        run_name = os.environ.get("BIAS_RUN_NAME")

        return Settings(
            db_url=db_url,
            model_dir=model_dir,
            encoder_name=encoder,
            max_length=max_length,
            max_chunks=max_chunks,
            run_name=run_name,
        )