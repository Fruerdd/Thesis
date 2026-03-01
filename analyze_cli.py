import argparse
import json
from pathlib import Path

from db_pg import make_session_factory
from analyze_and_store import (
    ensure_model_config,
    analyze_text_and_store,
    analyze_url_and_store,
)

# IMPORTANT: keep these consistent with your training script
LEAN_CANON = ["Right", "Right-center", "Center", "Left-center", "Left"]
INT_CANON  = ["Highly Biased", "Neutral", "Slightly Biased"]
MAX_LENGTH = 256
MAX_CHUNKS = 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="postgresql+psycopg2://user:pass@host:5432/dbname")
    ap.add_argument("--model_dir", required=True, help="path to trained student model folder")
    ap.add_argument("--encoder", required=True, help="encoder name, e.g. bert-base-uncased")
    ap.add_argument("--source", required=True, help="news source name (BBC, Fox, etc.)")
    ap.add_argument("--threshold", type=float, default=0.5)

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, default=None)
    g.add_argument("--url", type=str, default=None)

    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--notes", type=str, default=None)

    args = ap.parse_args()

    Session = make_session_factory(args.db)
    with Session() as session:
        model_config_id = ensure_model_config(
            session,
            model_dir=args.model_dir,
            encoder_name=args.encoder,
            max_length=MAX_LENGTH,
            max_chunks=MAX_CHUNKS,
            lean_labels=LEAN_CANON,
            intensity_labels=INT_CANON,
            run_name=args.run_name,
            notes=args.notes,
            train_summary=None,
        )

    if args.text is not None:
        row_id = analyze_text_and_store(
            db_url=args.db,
            model_config_id=model_config_id,
            source_name=args.source,
            text=args.text,
            encoder_name=args.encoder,
            model_dir=args.model_dir,
            threshold=args.threshold,
        )
        print(json.dumps({"saved_article_analysis_id": row_id}, indent=2))

    if args.url is not None:
        row_id = analyze_url_and_store(
            db_url=args.db,
            model_config_id=model_config_id,
            source_name=args.source,
            url=args.url,
            encoder_name=args.encoder,
            model_dir=args.model_dir,
            threshold=args.threshold,
        )
        print(json.dumps({"saved_article_analysis_id": row_id}, indent=2))


if __name__ == "__main__":
    main()