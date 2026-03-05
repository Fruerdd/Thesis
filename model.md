# Thesis Project Overview

## What this project is

This project is a thesis-focused NLP system for political bias analysis of news content. It is designed to analyze news in two related dimensions:

1. **Political leaning**
   - Right
   - Right-center
   - Center
   - Left-center
   - Left

2. **Bias intensity**
   - Highly Biased
   - Neutral
   - Slightly Biased

The current implementation combines model training, pseudo-label generation, inference, article analysis workflows, and storage of analysis results in PostgreSQL.

---

## Core use cases

The system currently supports two main analysis flows after training:

1. **Analyze from raw text + source name**
   - You provide article text directly.
   - You also provide the news source name (for example, BBC, CNN, Fox News).
   - The model predicts political leaning, intensity, and a derived bias score.
   - The result can be stored in the database.

2. **Analyze from URL + source name**
   - You provide a URL for a news article.
   - The system is expected to fetch and extract article content.
   - You also provide the source name.
   - The model runs the same prediction flow on the extracted text.
   - The result can be stored in the database.

The source name is important because it is a key contextual attribute you explicitly wanted saved for each analyzed article.

---

## High-level architecture

The project is organized around five logical layers:

### 1. Data layer
Uses two datasets during training:

- **AllSides headlines dataset**
  - Used for supervised **leaning** labels.
- **NewsMediaBias full articles dataset**
  - Used for supervised **intensity** labels.

This means the project is a multi-task system where one task has strong headline supervision and the other has strong full-article supervision.

### 2. Pseudo-labeling layer
Because the article dataset does not provide native leaning labels, the project uses a **teacher-student setup**:

- A **teacher model** is trained on headlines for leaning.
- The teacher then predicts soft leaning probabilities for full articles.
- These soft predictions become **pseudo-labels** for the student model.
- A confidence threshold is used so low-confidence pseudo-labels can be ignored.

### 3. Student multi-task model
The final model is a multi-task hierarchical model that learns both:

- political leaning
- bias intensity

It uses hard labels where available and soft pseudo-labels where direct labels are missing.

### 4. Inference layer
A prediction function accepts text and produces:

- predicted political leaning
- predicted bias intensity
- biased / not biased flag
- bias score
- class probabilities
- chunk attention weights

### 5. Persistence layer
PostgreSQL is used to store:

- model configuration metadata
- every analyzed article result

This makes the project suitable for experimentation, auditing, and future dashboards or APIs.

---

## Model design

### Base encoder
The project uses a Transformer encoder:

- Default encoder: **`bert-base-uncased`**

This encoder processes tokenized text chunks and produces contextual embeddings.

### Hierarchical chunking
Articles can be longer than a single transformer window, so the system splits text into chunks.

Current configuration:

- `MAX_LENGTH = 128`
- `MAX_CHUNKS = 3`

How it works:

1. Text is tokenized without adding special tokens.
2. Tokens are sliced into chunk-sized segments.
3. Each chunk gets `[CLS] ... [SEP]`.
4. Chunks are padded to fixed length.
5. A `chunk_mask` marks which chunks are real and which are padding.

This lets the model process more context than one single BERT pass while keeping memory controlled.

### Handcrafted feature branch
In addition to transformer embeddings, the model computes lightweight stylistic features from text, including:

- word count
- character count
- average word length
- long-word ratio
- exclamation/question counts
- quote counts
- uppercase ratio
- punctuation ratio
- hedge-word ratio
- intensifier ratio
- negation ratio

These features are projected into the same hidden space as the encoder output and fused with the document representation.

### Chunk attention
A learned attention layer scores each chunk and computes a weighted document vector.

This means the model does not treat every chunk equally - it learns which chunk matters more for the current document representation.

### Task-specific fusion
The model uses two separate gating paths:

- one gate for **leaning**
- one gate for **intensity**

This allows each task to combine:

- transformer-derived document context
- handcrafted feature representation

in a different way.

### Output heads
The model has three heads:

1. **Lean head**
   - 5-class classification
2. **Intensity head**
   - 3-class classification
3. **Domain head**
   - predicts headline vs article domain

### Domain adversarial component
The project includes a gradient reversal layer (GRL) for domain adaptation.

Current recommended setting in your code:

- `W_DOMAIN = 0.0`

This effectively disables the domain-loss contribution in training for the current run, which is useful because strong domain suppression was one of the likely reasons for weaker leaning performance.

---

## Teacher-student training strategy

### Teacher stage
The teacher is trained only on the headline dataset for political leaning.

Purpose:

- create a cleaner leaning signal
- learn the lean classes before the student mixes tasks

Teacher outputs:

- a trained leaning model
- soft leaning probabilities for article texts

### Pseudo-label stage
The teacher runs inference on article texts and writes pseudo-label probabilities to cache.

Stored per article row:

- `row_id`
- `p0` through `p4` (lean probabilities)

Pseudo-labels are filtered by confidence using:

- `PSEUDO_MIN_CONF = 0.45`

If the teacher is not confident enough, the article gets no soft leaning supervision.

### Student stage
The student is trained on merged data:

- **Headlines**: hard leaning labels
- **Articles**: hard intensity labels
- **Articles**: soft leaning pseudo-labels (if confidence threshold is met)

This is the main training stage for the final deployed model.

---

## Current training configuration

### Hardware-aware behavior
The script supports:

- CUDA (NVIDIA GPU)
- MPS (Apple Silicon GPU)
- CPU fallback

Automatic mixed precision is enabled on:

- CUDA
- MPS

GradScaler is used only on CUDA.

### Current defaults

- `BATCH_SIZE = 8` on CUDA
- `BATCH_SIZE = 4` otherwise
- `GRAD_ACCUM = 2`
- effective batch is approximately 16
- `EPOCHS_TEACHER = 2`
- `EPOCHS_STUDENT = 3`
- `LR = 2e-5`
- `WARMUP_RATIO = 0.06`

### Lean-friendly loss weights
Current recommended weights in your code:

- `W_LEAN_HARD = 1.5`
- `W_LEAN_SOFT = 0.5`
- `W_INTENSITY = 1.0`
- `W_DOMAIN = 0.0`

Why this matters:

- hard leaning labels are currently the most reliable lean signal, so they are weighted more heavily
- pseudo-labels help, but are intentionally weaker because they are noisier than ground truth
- intensity remains fully supervised on articles
- domain loss is disabled for now because it can suppress useful domain-specific signals needed for leaning

---

## Why leaning is the harder task

The project currently has a built-in asymmetry:

- **Leaning** is supervised primarily from headlines.
- **Intensity** is supervised directly from full articles.

That creates several challenges:

1. **Label mismatch across domains**
   - Leaning is learned from headline style.
   - At inference time, the model often needs to judge full articles.

2. **Pseudo-label noise**
   - Article leaning labels are not real ground-truth labels.
   - They are teacher predictions, so they can be wrong or overly uncertain.

3. **Class overlap**
   - The five lean classes are much closer to each other than the three intensity classes.
   - Especially around `Center`, `Right-center`, and `Left-center`.

4. **Intensity is easier structurally**
   - Intensity often correlates with clearer lexical and stylistic cues.
   - Leaning is more subtle and more context-dependent.

So it is normal in the current version that:

- **intensity metrics are much higher**
- **lean accuracy / macro F1 are lower**

---

## Dataset handling

### Separate sampling strategy
One of the important recent fixes is that student data is no longer sampled as one combined pool.

Instead:

- headlines are sampled separately and stratified by `y_lean`
- articles are sampled separately and stratified by `y_int`

This prevents the student dataset from becoming unintentionally skewed toward one task or one domain.

### Current test-size reduction
The project currently supports running on only a fraction of the student data for debugging and faster validation.

Current setting:

- `STUDENT_DATA_FRACTION = 0.5`

That means student training uses about 50% of the original trainable data after the per-task stratified sampling step.

This is useful when you want to:

- validate the pipeline quickly
- confirm loss behaves correctly
- verify database integration and end-to-end flow
- avoid very long runs while tuning architecture

---

## Caching and memory strategy

### Why caching exists
Pseudo-labeling the article dataset can be expensive, so the project stores teacher outputs to disk.

Cache locations:

- `articles_pseudo_lean.parquet`
- `articles_pseudo_lean.csv.gz`
- `pseudo_parts/part_*.parquet`

### Why chunked pseudo caches exist
Appending directly to a single parquet file is awkward, so chunk files may be created under `pseudo_parts/`.

### Important fix
A recent critical fix is that cache loading must read:

- the main parquet file
- all parquet parts
- the CSV fallback if present

and then concatenate and deduplicate by `row_id`.

If this is not done, a large portion of pseudo-labels can silently disappear, which directly harms student leaning performance.

### Clear-cache support
The training script includes a helper to clear pseudo-label cache before regenerating it.

This is useful when:

- you change teacher weights
- you change chunk settings
- you change confidence threshold
- a previous pseudo-label run was incomplete

---

## Inference output

The `predict()` path returns:

- `political_bias`
- `bias_intensity`
- `biased`
- `biased_score`
- `probs_lean`
- `probs_int`
- `chunk_attention`

### Bias score logic
`biased_score` is computed as:

- `1.0 - P(Center)`

So the more confident the model is that a text is non-center, the higher the bias score.

### Thresholding
The system then converts this into a boolean:

- `biased = biased_score >= threshold`

Default threshold:

- `0.5`

---

## PostgreSQL integration

The project is designed to store analysis metadata and results in PostgreSQL.

### Intended database structure
You wanted two core tables.

#### 1. Model configuration table
Purpose:

- store the identity and settings of a trained model version
- make results reproducible and auditable

Recommended fields:

- `id`
- `model_name`
- `encoder_name`
- `model_dir`
- `max_length`
- `max_chunks`
- `batch_size`
- `grad_accum`
- `epochs_teacher`
- `epochs_student`
- `lr`
- `warmup_ratio`
- `w_lean_hard`
- `w_lean_soft`
- `w_intensity`
- `w_domain`
- `pseudo_min_conf`
- `student_data_fraction`
- `created_at`
- optional notes / description

#### 2. Article analysis table
Purpose:

- store every analyzed article or text request
- preserve source metadata and predictions

Recommended fields:

- `id`
- `model_config_id` (FK)
- `source_name`
- `input_mode` (`text` or `url`)
- `url` (nullable)
- `article_text`
- `article_text_hash` (optional dedupe)
- `political_bias`
- `bias_intensity`
- `biased`
- `biased_score`
- `prob_right`
- `prob_right_center`
- `prob_center`
- `prob_left_center`
- `prob_left`
- `prob_highly_biased`
- `prob_neutral`
- `prob_slightly_biased`
- `chunk_attention_json`
- `created_at`

### Why source name matters
You explicitly wanted the news source name saved for each analyzed item.

That is valuable because it allows you to:

- group predictions by publisher
- compare source-level distributions
- audit model behavior by brand/source
- build dashboards later

---

## CLI workflow

The current workflow is intended to be run in stages:

### 1. Train teacher
```bash
python train_bias_v2.py --train_teacher
```

### 2. Generate pseudo-labels
```bash
python train_bias_v2.py --pseudo_label_articles
```

### 3. Train student
```bash
python train_bias_v2.py --train_student
```

### 4. Predict from text
```bash
python train_bias_v2.py --predict "Some news text here"
```

### 5. Analyze and store via separate CLI
A separate script (for example `analyze_cli.py`) can:

- connect to PostgreSQL
- run prediction using the trained student model
- insert a result row into the database

---

## Known weak points / current limitations

### 1. Leaning labels are still the biggest bottleneck
The strongest limitation remains:

- no native article-level leaning ground truth

That means article leaning is still learned indirectly.

### 2. Pseudo-label quality controls final lean quality
If the teacher is mediocre, the student inherits some of that error.

### 3. `MAX_LENGTH = 128` is a speed-first compromise
It helps runtime and memory, but it reduces the amount of local context visible inside each chunk.

### 4. Long training time
Student training can still be very long, especially on large merged datasets.

### 5. Source name is external metadata
The model does not currently use source name as an input feature for prediction - it is stored as metadata, not modeled as a supervised signal.

---

## Recommended next steps

### Model quality
1. Add a confusion matrix for the leaning task.
2. Print per-class precision/recall/F1 for all 5 lean classes.
3. Tune `PSEUDO_MIN_CONF` (for example 0.45, 0.55, 0.65).
4. Compare runs with:
   - `MAX_CHUNKS = 2`
   - `MAX_CHUNKS = 3`
   - `MAX_LENGTH = 128` vs `160` vs `192`
5. Try one run with the encoder frozen for the first epoch, then fully unfrozen.

### Data quality
1. Add a true article-level leaning dataset if possible.
2. Reduce noisy pseudo-labels more aggressively if teacher confidence is low.
3. Inspect class balance per lean label after downsampling.

### Productization
1. Add a robust URL article extractor.
2. Add deduplication by URL or text hash before storing results.
3. Add an API layer (FastAPI / Flask) for external calls.
4. Add a dashboard to visualize:
   - results by source
   - source-level bias distributions
   - time-based trend shifts

---

## Current project status

At the current stage, the project is a functioning end-to-end experimental pipeline that already includes:

- hierarchical BERT-based multi-task classification
- teacher-student pseudo-labeling for article leaning
- hardware-aware training for CUDA / MPS / CPU
- memory-safe student training via streaming-style DataLoader usage
- text prediction pipeline
- PostgreSQL-ready storage design for model configs and article analyses
- support for source-aware result storage

In short, this is no longer just a model training script - it is evolving into a full **news bias analysis platform prototype** for thesis work, experimentation, and potential application integration.
