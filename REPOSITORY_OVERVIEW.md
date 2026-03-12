# Repository Overview

## Purpose

This repository contains a thesis project for political bias analysis of news content. The codebase combines:

- model training for political leaning and bias intensity
- pseudo-label generation for weakly supervised learning
- evaluation scripts for local and batch testing
- inference utilities for direct text or URL-based analysis
- persistence layers for storing predictions in PostgreSQL
- a FastAPI backend for serving the trained model

At a high level, the repository is trying to solve two related NLP tasks:

1. Predict political leaning:
   - `Right`
   - `Right-center`
   - `Center`
   - `Left-center`
   - `Left`
2. Predict bias intensity:
   - `Highly Biased`
   - `Neutral`
   - `Slightly Biased`

## Current Shape Of The Repository

The repository has grown in layers. There are three main generations of code:

1. Early single-task BERT training scripts
2. A first multitask hierarchical model (`multitask_bias_full.py`)
3. A newer teacher-student multitask pipeline (`train_bias_v2.py`) plus serving code

The most important current components are:

- `train_bias_v2.py`: main training and inference implementation
- `finetune_lean_lrll.py`: post-training lean-focused fine-tuning
- `backend/`: API server and DB-backed inference flow
- `analyze_cli.py` and `analyze_and_store.py`: older CLI/storage workflow
- `eval_political_bias_local.py`, `batch_eval_datasets.py`, `roc_eval.py`: evaluation utilities

## Repository Structure

### Top-level training and inference

- `train_bias_v2.py`
  - Main training pipeline for the current model.
  - Implements:
    - label normalization
    - handcrafted text features
    - hierarchical chunk encoding
    - teacher model training
    - pseudo-label generation for articles
    - student multitask training
    - model save/load
    - runtime prediction
- `finetune_lean_lrll.py`
  - Additional fine-tuning stage focused on improving leaning predictions.
  - Mixes newer political bias datasets with replay data from AllSides to reduce forgetting.
- `multitask_bias_full.py`
  - Older multitask hierarchical model implementation.
  - Useful as a historical baseline, but it appears to be superseded by `train_bias_v2.py`.
- `main.py`
  - Separate simpler BERT training script for article/headline classification experiments.
  - Looks more like an earlier standalone training path than the current primary pipeline.
- `predict.py`
  - Standalone 5-class political leaning training/inference script using `AutoModelForSequenceClassification`.
  - Simpler than the multitask system.
- `test.py`
  - Minimal local inference helper for a saved single-task model.

### Root-level analysis and persistence workflow

- `analyze_and_store.py`
  - Runs inference and stores results into PostgreSQL.
  - Supports both raw text and URL-based article analysis.
- `analyze_cli.py`
  - Command-line wrapper around `analyze_and_store.py`.
- `db_pg.py`
  - SQLAlchemy ORM models and session factory for the root-level DB workflow.
- `url_extract.py`
  - Fetches article HTML and extracts title/body text using `requests` and `BeautifulSoup`.

### Backend API

The `backend/` directory is a more structured serving layer:

- `backend/app.py`
  - FastAPI application.
  - Exposes `/health` and `/analyze`.
  - Accepts either text or a URL in one field, runs extraction if needed, stores the result in PostgreSQL, and returns the saved analysis.
- `backend/infer.py`
  - Safe wrapper around `train_bias_v2.predict()`.
  - Converts NumPy/PyTorch values into JSON-safe Python types.
- `backend/extract.py`
  - URL fetching and HTML-to-text extraction for the API path.
- `backend/db.py`
  - Session factory and model configuration lookup/creation.
- `backend/models.py`
  - SQLAlchemy ORM models for `model_configs` and `article_analyses`.
- `backend/schema.py`
  - Pydantic request/response models.
- `backend/settings.py`
  - Loads runtime configuration from environment variables like `BIAS_DB_URL`, `BIAS_MODEL_DIR`, and `BIAS_ENCODER`.

This backend is the cleanest serving path in the repository.

### Evaluation and dataset processing

- `eval_political_bias_local.py`
  - Evaluates leaning predictions on a local CSV test set.
  - Produces confusion matrix, accuracy, and classification report.
- `batch_eval_datasets.py`
  - Runs the model over folders of JSON article datasets and writes CSV outputs.
- `roc_eval.py`
  - Builds ROC curves and AUC summaries for lean and intensity outputs.
- `batch_outputs_finetuned/`, `batch_outputs_lrll/`, `cmp_old/`
  - Stored batch evaluation outputs and comparisons.

### Data and artifacts

- `data/`
  - Training datasets used by the experiments.
  - Includes AllSides, NewsMediaBias, and political bias CSV files.
- `data_test/`
  - Local evaluation data.
- `bias_system_v2/`
  - Output directory for the current multitask teacher-student system.
  - Stores checkpoints, pseudo-label cache, and metadata.
- `How_to_train`
  - Informal command reference for training, evaluation, and analysis flows.
- `model.md`
  - Existing model-focused documentation.

## Main Modeling Approach

The current model is implemented in `train_bias_v2.py` and uses a teacher-student multitask design.

### Why teacher-student exists

The repository combines datasets with incomplete supervision:

- headline data provides strong leaning labels
- full-article data provides intensity labels
- article-level leaning labels are weaker or missing in the main pipeline

To bridge that gap:

1. A teacher model is trained on headline leaning labels.
2. The teacher predicts soft leaning probabilities for full articles.
3. Those soft predictions become pseudo-labels.
4. A student model is trained on:
   - hard leaning labels from headlines
   - hard intensity labels from articles
   - soft leaning pseudo-labels for articles

### Model architecture

The `v2` model is not a plain single-pass BERT classifier. It includes:

- a transformer encoder (`bert-base-uncased` by default)
- hierarchical chunking for long texts
- handcrafted stylistic features
- attention over chunks
- separate heads for:
  - political leaning
  - bias intensity
  - domain classification

There is also domain-adversarial support through gradient reversal, although the current defaults set domain loss weight to `0.0`, effectively disabling that part during the recommended run.

### Long-text handling

Long articles are split into fixed-size token chunks. Each chunk is encoded separately, then aggregated with learned chunk attention. This avoids relying on a single BERT window for long documents.

### Handcrafted feature branch

The model augments transformer embeddings with shallow text statistics such as:

- word count
- character count
- punctuation ratios
- uppercase ratio
- hedge/intensifier/negation ratios

These features are fused into the final document representation before classification.

## End-To-End Workflow

### Training workflow

The intended `v2` sequence, documented in `How_to_train`, is:

1. `python train_bias_v2.py --train_teacher`
2. `python train_bias_v2.py --pseudo_label_articles`
3. `python train_bias_v2.py --train_student`

Optional follow-up:

4. `python finetune_lean_lrll.py ...`

This extra fine-tuning stage focuses on leaning quality by training further on political bias CSV datasets plus replay samples from AllSides.

### Inference workflow

There are two main ways to use a trained model:

1. CLI/database path
   - `analyze_cli.py`
   - `analyze_and_store.py`
2. API path
   - `backend/app.py`

Both workflows can:

- analyze plain text
- analyze a URL after extracting article text
- save predictions and metadata into PostgreSQL

### Stored output

Predictions typically include:

- `political_bias`
- `bias_intensity`
- `biased`
- `biased_score`
- class probabilities
- chunk attention weights

The DB schemas also store:

- source name
- URL or text mode
- extracted title/text
- fetch errors / HTTP status
- model configuration used
- request duration

## Database Design

There are two parallel DB implementations:

- `db_pg.py` for the older root-level workflow
- `backend/models.py` plus `backend/db.py` for the API workflow

Both center around the same idea:

- `model_configs`
  - metadata about the trained model and tokenizer configuration
- `article_analyses`
  - one row per analyzed article or text submission

The backend version is cleaner and more aligned with the current API.

## Important Datasets

The repository uses several datasets for different purposes:

- `data/allsides_balanced_news_headlines-texts.csv`
  - leaning supervision for headline data
- `data/newsmediabias-full.csv`
  - article data with bias/intensity-related labels
- `data/Political_Bias.csv`
  - additional leaning-focused fine-tuning data
- `data/Political_Bias_Update.csv`
  - additional leaning-focused fine-tuning data
- `data/allsides.csv`
  - source-to-bias mapping and related metadata used in older scripts
- `data_test/political_bias.csv`
  - local evaluation set used by `eval_political_bias_local.py`

## What Looks Current Versus Legacy

### Most current path

If you want to understand the repository as it exists now, start with:

1. `train_bias_v2.py`
2. `finetune_lean_lrll.py`
3. `backend/app.py`
4. `backend/infer.py`
5. `backend/models.py`

That combination represents the main training-to-serving story.

### Likely legacy or experimental paths

These still matter for context, but they look secondary:

- `main.py`
- `predict.py`
- `test.py`
- `multitask_bias_full.py`
- root-level DB flow in `analyze_cli.py` / `analyze_and_store.py`

They appear to reflect earlier stages of the project or alternative experiments that were kept in the repository.

## Dependencies

The backend dependency file is `requirements-backend.txt`, which includes:

- `fastapi`
- `uvicorn[standard]`
- `sqlalchemy`
- `psycopg2-binary`
- `pydantic`
- `requests`
- `trafilatura`
- `python-dotenv`

The training scripts also rely on packages that are not listed there, including:

- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `tqdm`
- optionally `captum`
- `beautifulsoup4`

So the repository does not currently appear to have one complete dependency lockfile for all training and serving paths.

## Practical Reading Order

For someone new to the codebase, the fastest way to understand it is:

1. Read `model.md` for the conceptual model summary.
2. Read `train_bias_v2.py` for the real training and inference implementation.
3. Read `How_to_train` for the expected command sequence.
4. Read `backend/app.py` and `backend/infer.py` for deployed usage.
5. Read `finetune_lean_lrll.py` for the post-training refinement step.
6. Use the evaluation scripts to understand how performance is checked.

## Summary

This repository is a thesis-oriented news bias analysis system built around a hierarchical BERT-based multitask classifier. The central idea is to learn political leaning and bias intensity jointly, using teacher-generated pseudo-labels to compensate for incomplete article-level supervision. Around that model, the repository includes evaluation tooling, PostgreSQL persistence, a CLI workflow, and a newer FastAPI backend for serving predictions.

The repository is functional but layered: some files represent old experiments, while the clearest current path is `train_bias_v2.py` plus the `backend/` service.
