# Thesis1: Political Bias Detection for News Articles

This repository contains a thesis project for detecting political bias in news content. It includes training pipelines, dataset preparation utilities, evaluation scripts, inference utilities, PostgreSQL-backed storage, and a FastAPI backend for serving predictions.

The project covers two related prediction tasks:

- political leaning:
  - `Right`
  - `Right-center`
  - `Center`
  - `Left-center`
  - `Left`
- bias intensity:
  - `Highly Biased`
  - `Neutral`
  - `Slightly Biased`

## What The Project Can Do

The repository currently supports these feature areas:

- train hierarchical Transformer-based models for political leaning and bias intensity
- generate pseudo-labels for article-level learning
- fine-tune leaning performance on additional political bias datasets
- apply source priors for ambiguous leaning predictions in the `v3` pipeline
- evaluate models on local holdout and external datasets
- generate confusion matrices, reports, ROC curves, and batch CSV outputs
- analyze raw article text or a news URL
- extract article text from URLs
- store inference results and model metadata in PostgreSQL
- serve predictions through a FastAPI API

## Main Pipelines

### `v2` multitask teacher-student pipeline

Implemented mainly in `train_bias_v2.py`.

Core ideas:

- a teacher model learns political leaning from headline data
- the teacher generates soft pseudo-labels for full articles
- a student model learns:
  - hard leaning labels from headline data
  - hard intensity labels from article data
  - soft leaning labels for articles
- the student uses:
  - `bert-base-uncased` by default
  - hierarchical chunking for long documents
  - handcrafted stylistic features
  - chunk attention
  - multitask heads for leaning and intensity

This is the main multitask path used by the backend and CLI inference flow.

### `v3` leaning-focused pipeline

Implemented mainly in `train_bias_v3.py`.

Additional features in `v3`:

- training on prepared combined leaning datasets from `data_prepared/`
- source-bias priors from `data/allsides.csv`
- ambiguity-aware source prior application
- stronger focus on 5-class leaning quality
- teacher, student, and combined evaluation modes

This is the newest leaning-oriented experimentation path in the repository.

## Repository Layout

### Training and modeling

- `train_bias_v2.py`
  - current multitask training, pseudo-labeling, saving, loading, and prediction
- `train_bias_v3.py`
  - newer leaning-focused training and inference pipeline
- `finetune_lean_lrll.py`
  - post-training fine-tuning on additional political bias datasets with replay sampling
- `multitask_bias_full.py`
  - older multitask hierarchical model version
- `main.py`
  - earlier BERT training experiments
- `predict.py`
  - simpler standalone 5-class leaning model training/inference script
- `test.py`
  - minimal local inference helper for a saved single-task model

### Dataset preparation

- `build_combined_lean_dataset.py`
  - combines multiple lean-labeled datasets into one cleaned dataset
- `build_real_holdout_and_train_split.py`
  - builds a balanced real-only holdout and the remaining train split
- `build_5class_100_each_from_combined.py`
  - creates a balanced 5-class evaluation set
- `build_balanced_synthetic_lean_dataset.py`
  - prepares synthetic balanced leaning data

### Evaluation

- `eval_political_bias_local.py`
  - evaluates local CSV leaning datasets against a saved model
- `eval_new_5class_dataset.py`
  - evaluates the `v3` pipeline on balanced 5-class datasets
- `eval_lean_external.py`
  - evaluates 5-class performance on external datasets
- `eval_lean_external_3class.py`
  - evaluates external data collapsed into 3 classes
- `batch_eval_datasets.py`
  - runs batch inference over nested JSON datasets and writes CSV outputs
- `roc_eval.py`
  - generates ROC curves and AUC summaries

### Inference, storage, and API

- `analyze_cli.py`
  - CLI entrypoint for analyzing text or URLs and storing results
- `analyze_and_store.py`
  - root-level storage workflow for inference results
- `db_pg.py`
  - SQLAlchemy models and session factory for the root-level DB flow
- `url_extract.py`
  - root-level URL fetching and extraction
- `backend/app.py`
  - FastAPI server with `/health` and `/analyze`
- `backend/infer.py`
  - JSON-safe inference wrapper over `train_bias_v2.predict()`
- `backend/extract.py`
  - backend URL extraction
- `backend/models.py`
  - backend ORM models for `model_configs` and `article_analyses`
- `backend/db.py`
  - backend DB helpers
- `backend/settings.py`
  - backend environment configuration

### Data and outputs

- `data/`
  - source training datasets
- `data_prepared/`
  - processed and merged leaning datasets
- `data_test/`
  - local evaluation datasets
- `Datasets/`
  - JSON dataset folders used for batch evaluation
- `bias_system_v2/`
  - saved outputs for the `v2` pipeline
- `bias_system_v3/`
  - saved outputs for the `v3` pipeline
- `batch_outputs_finetuned/`, `batch_outputs_lrll/`, `roc_outputs/`
  - generated evaluation artifacts

## Model Design

The main hierarchical models in this repository use:

- a Transformer encoder, typically `bert-base-uncased`
- long-text chunking instead of a single-pass input window
- learned chunk attention
- handcrafted features such as:
  - word count
  - character count
  - punctuation ratios
  - uppercase ratio
  - hedge/intensifier/negation signals
- separate output heads for different tasks

In `v2`, the model predicts both leaning and intensity. In `v3`, the main emphasis shifts toward higher-quality leaning prediction and better use of source-level prior information.

## Datasets Used

Important datasets referenced in the codebase include:

- `data/allsides_balanced_news_headlines-texts.csv`
  - headline leaning supervision
- `data/newsmediabias-full.csv`
  - article data with bias/intensity-related supervision
- `data/allsides.csv`
  - source-to-bias mapping for priors and metadata
- `data/Political_Bias.csv`
  - additional leaning fine-tuning data
- `data/Political_Bias_Update.csv`
  - additional leaning fine-tuning data
- `data_prepared/combined_lean_train_without_holdout.csv`
  - prepared leaning train split for `v3`
- `data_prepared/combined_lean_real_holdout_5class_100_each.csv`
  - balanced real holdout for evaluation
- `data_test/political_bias.csv`
  - local evaluation CSV

## Typical Workflows

### 1. Train the `v2` multitask model

```bash
python train_bias_v2.py --train_teacher
python train_bias_v2.py --pseudo_label_articles
python train_bias_v2.py --train_student
```

Optional extra fine-tuning:

```bash
python finetune_lean_lrll.py \
  --model_dir "./bias_system_v2/student_mt_softlean_seed42" \
  --encoder "bert-base-uncased" \
  --out_dir "./bias_system_v2/student_finetuned_lrll_v2" \
  --epochs 2 \
  --batch_size 8 \
  --max_length 192 \
  --max_chunks 3 \
  --replay_size 12000 \
  --lrll_repeat 4
```

### 2. Train the `v3` leaning pipeline

```bash
python train_bias_v3.py --train_teacher --encoder bert-base-uncased
python train_bias_v3.py --clear_pseudo_cache
python train_bias_v3.py --pseudo_label_articles --encoder bert-base-uncased
python train_bias_v3.py --train_student --encoder bert-base-uncased
```

Example direct prediction:

```bash
python train_bias_v3.py \
  --predict "Your article text here" \
  --predict_mode combined \
  --predict_source "CNN" \
  --encoder bert-base-uncased
```

### 3. Evaluate models

Local leaning evaluation:

```bash
python eval_political_bias_local.py \
  --csv "data_test/political_bias.csv" \
  --model_dir "./bias_system_v2/student_mt_softlean_seed42" \
  --encoder "bert-base-uncased" \
  --max_length 192 \
  --max_chunks 3 \
  --batch_size 16
```

Balanced 5-class evaluation:

```bash
python eval_new_5class_dataset.py \
  --csv "data_prepared/combined_lean_real_holdout_5class_100_each.csv" \
  --mode teacher \
  --batch_size 32 \
  --out_json "real_holdout_eval_teacher.json" \
  --out_png "real_holdout_cm_teacher.png" \
  --out_png_norm "real_holdout_cm_teacher_norm.png" \
  --out_preds "real_holdout_preds_teacher.csv"
```

External 5-class evaluation:

```bash
python eval_lean_external.py \
  --csv "data_test/news_bias.csv" \
  --mode combined \
  --batch_size 32 \
  --out_json "news_bias_eval.json" \
  --out_png "news_bias_cm.png" \
  --out_png_norm "news_bias_cm_norm.png" \
  --out_preds "news_bias_predictions.csv"
```

External 3-class evaluation:

```bash
python eval_lean_external_3class.py \
  --csv "data_test/news_bias.csv" \
  --mode combined \
  --batch_size 32 \
  --out_json "news_bias_eval_3class.json" \
  --out_png "news_bias_cm_3class.png" \
  --out_png_norm "news_bias_cm_3class_norm.png" \
  --out_preds "news_bias_predictions_3class.csv"
```

ROC evaluation:

```bash
python roc_eval.py \
  --model_dir "./bias_system_v2/student_mt_softlean_seed42" \
  --encoder "bert-base-uncased" \
  --split both
```

Batch evaluation on JSON article folders:

```bash
python batch_eval_datasets.py \
  --datasets_root "Datasets" \
  --model_dir "./bias_system_v2/student_mt_softlean_seed42" \
  --encoder "bert-base-uncased" \
  --out_dir "batch_outputs" \
  --max_length 128 \
  --max_chunks 3 \
  --batch_size 16 \
  --threshold 0.5
```

### 4. Analyze text or a URL and save to PostgreSQL

Analyze plain text:

```bash
python analyze_cli.py \
  --db "postgresql+psycopg2://user:pass@localhost:5432/biasdb" \
  --model_dir "./bias_system_v2/student_mt_softlean_seed42" \
  --encoder "bert-base-uncased" \
  --source "BBC" \
  --text "Your article text here..."
```

Analyze URL:

```bash
python analyze_cli.py \
  --db "postgresql+psycopg2://user:pass@localhost:5432/biasdb" \
  --model_dir "./bias_system_v2/student_mt_softlean_seed42" \
  --encoder "bert-base-uncased" \
  --source "BBC" \
  --url "https://example.com/news/article"
```

### 5. Run the FastAPI backend

Set environment variables first:

```bash
export BIAS_DB_URL="postgresql+psycopg2://user:pass@localhost:5432/biasdb"
export BIAS_MODEL_DIR="./bias_system_v2/student_mt_softlean_seed42"
export BIAS_ENCODER="bert-base-uncased"
export BIAS_MAX_LENGTH="128"
export BIAS_MAX_CHUNKS="3"
```

Run the API:

```bash
uvicorn backend.app:app --reload
```

Available endpoints:

- `GET /health`
- `POST /analyze`

Request shape:

```json
{
  "source": "BBC",
  "link_or_text": "https://example.com/article-or-plain-text",
  "threshold": 0.5
}
```

The API:

- detects whether input is a URL or plain text
- extracts article content when needed
- runs inference
- stores the result in PostgreSQL
- returns the saved prediction record

## Database Model

The storage layer is centered around two entities:

- `model_configs`
  - model path, encoder, chunk settings, labels, run metadata
- `article_analyses`
  - source name, URL/text mode, extracted content, prediction JSON, bias outputs, timing

This allows the project to keep an audit trail of which model produced which prediction.

## Dependencies

`requirements-backend.txt` covers the API-side dependencies:

- `fastapi`
- `uvicorn[standard]`
- `sqlalchemy`
- `psycopg2-binary`
- `pydantic`
- `requests`
- `trafilatura`
- `python-dotenv`

Training and evaluation also require additional packages used directly in the scripts, including:

- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `tqdm`
- `beautifulsoup4`
- optionally `captum`

## Notes

- `v2` is the main multitask training and serving path.
- `v3` is the newer leaning-focused experimentation path.
- The repository includes legacy and experimental scripts alongside the main flows.
- `How_to_train` contains additional command examples used during development.

## Suggested Reading Order

If you are new to the project, start here:

1. `train_bias_v2.py`
2. `train_bias_v3.py`
3. `finetune_lean_lrll.py`
4. `backend/app.py`
5. `How_to_train`
6. the evaluation scripts for the workflow you care about
