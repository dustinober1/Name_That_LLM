# 🎯 Name That LLM: Competitive Kaggle Solution Repository

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Managed by uv](https://img.shields.io/badge/managed%20by-uv-orange.svg)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

Production-grade, leak-free machine learning framework for the Kaggle **[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)** competition.

---

## 📌 Competition Overview

- **Objective:** Authorship attribution across frontier LLMs. Given a short passage of text (40 to 250 words) on varied academic and creative topics, predict which specific model generated it.
- **Evaluation Metric:** **Balanced Accuracy** (unweighted mean recall across each of the 18 classes).
- **Format:** **Code Competition** (Internet disabled, submissions generated via notebook execution producing `submission.parquet`).
- **Target Classes (18 Models across 6 AI Labs):**
  - **OpenAI:** `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`
  - **Google:** `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`
  - **Anthropic:** `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`
  - **Qwen:** `qwen3.8-max`
  - **ZAI (GLM):** `glm-5.3`, `glm-5.3-flash`
  - **Moonshot AI:** `kimi-k3`

---

## 🏆 Current Leaderboard & Experiment Log

See [EXPERIMENTS.md](file:///Users/dustinober/Projects/Data_Science/Name_That_LLM/EXPERIMENTS.md) for full diagnostic breakdowns and error analysis.

| Version | Model Family | Key Architecture | Local CV (Bal Acc) | Public LB (Bal Acc) | Link |
| :-: | :--- | :--- | :---: | :---: | :---: |
| **v1** | Linear Baseline | TF-IDF (Word + Char) + Stylometry + Logistic Regression (`balanced`) | **0.3245** | **0.2870** | [Version 1 on Kaggle](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline?scriptVersionId=359528236) |
| **v2-exp** | Standalone GBDT | 54 Stylometric features + 128 SVD components (LightGBM) | **0.2715** | — | — |
| **v2-blend** | **Ensemble Blend** | **85% Linear Baseline + 15% LightGBM (128 SVD + Stylometry)** | **`0.3282`** | **`0.3010`** | [Version 2 on Kaggle](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline) |

---

## ⚡ Core Challenge & Critical Invariant

> [!IMPORTANT]
> **Prompt Leakage Prevention**: In the test dataset, the passages are responses to **unseen prompts**. If cross-validation randomly splits samples, models will memorize prompt topic keywords (e.g. quantum physics vs Shakespeare) rather than authorship style.
> 
> **Every pipeline in this repo uses `StratifiedGroupKFold` grouped on `prompt_id`**, guaranteeing 100% zero prompt leakage between training and validation folds.

---

## 📁 Repository Structure

```tree
Name_That_LLM/
├── data/
│   ├── raw/                  # Official competition data (train.csv, test.csv)
│   ├── processed/            # Out-of-fold predictions, precomputed features
│   ├── sample/               # Synthetic mock dataset for offline testing & CI
│   └── external/             # Synthetic generation or public benchmark texts
├── src/name_that_llm/        # Core reusable package
│   ├── config.py             # Target classes, provider mappings, paths
│   ├── data.py               # Data loaders & StratifiedGroupKFold splitters
│   ├── metrics.py            # Balanced Accuracy, per-class recall, log-loss
│   ├── submission.py         # Strict schema validator & submission.parquet creator
│   ├── ensemble.py           # Multi-model probability blender (SLSQP / Nelder-Mead)
│   ├── features/
│   │   ├── stylometry.py     # Punctuation rates, discourse markers, syntax, TTR
│   │   └── tfidf.py          # Sublinear Word (1-2) + Char (3-5) n-gram FeatureUnion
│   ├── models/
│   │   ├── baseline_linear.py    # Class-weighted Logistic / Ridge / SGD pipelines
│   │   ├── gbm_classifier.py     # LightGBM with Stylometry + SVD features
│   │   └── transformer_classifier.py # PyTorch HuggingFace wrapper (MPS/CUDA)
│   └── evaluation/
│       └── cv_evaluator.py   # Leak-free CV engine with rich console reporting
├── scripts/
│   ├── download_data.py      # Kaggle CLI download helper
│   ├── generate_sample_data.py # Mock dataset builder
│   ├── train_baseline.py     # Train linear baseline (TF-IDF + Stylometry)
│   ├── train_lgbm.py         # Train LightGBM gradient boosted trees
│   ├── train_transformer.py  # Fine-tune Transformer backbones (e.g., DeBERTa)
│   └── build_submission_notebook.py # Builds standalone Kaggle notebook
├── notebooks/
│   ├── 01_eda_and_stylometry.ipynb  # EDA & feature analysis
│   └── kaggle_submission.ipynb      # Self-contained Kaggle submission notebook
├── submissions/              # Generated submission.parquet and submission.csv
├── tests/                    # Unit tests for CV leakage, metrics, and schema
└── pyproject.toml            # Fast environment & dependency management via uv
```

---

## 🚀 Quickstart

### 1. Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast and deterministic environment management.

```bash
# Sync dependencies and create virtual environment
uv sync
```

### 2. Downloading the Kaggle Competition Data

Make sure you have accepted the competition rules on [Kaggle](https://www.kaggle.com/competitions/name-that-llm/rules).

If you have a Kaggle API key (`kaggle.json`):
```bash
# Place your kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Download official train.csv and test.csv into data/raw/
uv run python scripts/download_data.py
```

*Note: If you haven't downloaded the real data yet, a mock dataset is already initialized in `data/sample/`, so all training scripts and tests run immediately!*

---

## 🔬 Training & Evaluation

### A. Train TF-IDF + Stylometry Baseline
Trains a 5-fold `StratifiedGroupKFold` linear model (TF-IDF word & char n-grams + hand-crafted stylometric features):
```bash
uv run python scripts/train_baseline.py --folds 5 --c 2.0
```

### B. Train LightGBM Classifier
Extracts stylometric vectors combined with 128 SVD components from TF-IDF:
```bash
uv run python scripts/train_lgbm.py --folds 5 --n-estimators 200 --lr 0.05
```

### C. Fine-Tune Transformers (Apple Silicon MPS / CUDA)
Fine-tunes a transformer model (e.g. `microsoft/deberta-v3-small` or `FacebookAI/roberta-base`) with class-weighted Cross-Entropy loss:
```bash
uv run python scripts/train_transformer.py --model-name microsoft/deberta-v3-small --epochs 3 --batch-size 8
```

---

## 📤 Submitting to Kaggle

Because **Name That LLM** is a **Code Competition**, predictions cannot be uploaded as a standalone file; they must be generated by a Kaggle Notebook running offline against the hidden test dataset.

### Option 1: Automated 1-Command CLI Submission (Recommended)
This script bundles the notebook, pushes it to Kaggle, monitors execution until complete, and submits the resulting `submission.parquet`:
```bash
uv run python scripts/submit_to_kaggle.py -m "TF-IDF + Stylometry 5-fold baseline"
```

### Option 2: Step-by-Step Kaggle CLI Commands
If you prefer running individual CLI commands:
```bash
# 1. Push the notebook to Kaggle (pre-configured in kaggle_kernel/)
kaggle kernels push -p kaggle_kernel

# 2. Check execution status until it reports 'Complete'
kaggle kernels status dustinober/name-that-llm-baseline

# 3. Submit the generated submission.parquet to the competition
kaggle competitions submit -c name-that-llm -f submission.parquet -k dustinober/name-that-llm-baseline -m "Baseline TF-IDF and Stylometry"
```

### Option 3: Browser UI Upload
1. Build the standalone notebook: `uv run python scripts/build_submission_notebook.py`
2. Navigate to [Name That LLM - Code](https://www.kaggle.com/competitions/name-that-llm/code) -> **"New Notebook"**
3. Select **File -> Upload Notebook** and choose `notebooks/kaggle_submission.ipynb`
4. In Settings: set **Internet: Off**
5. Click **"Save Version"** (Quick Save or Run All) and click **"Submit"**

- `model`: predicted model string from the 18 valid labels

---

## 🧪 Testing

Run the test suite to verify metric correctness, zero prompt leakage, and strict submission constraints:
```bash
uv run pytest -v
```

---

## 🏆 Key Strategies for Winning

1. **Stylometric Fingerprinting**: 
   - Punctuation rates (em-dashes, semicolons, backticks) are highly characteristic for labs like Anthropic vs OpenAI.
   - Sentence length variance, discourse transition patterns, and markdown formatting (lists, headers, tables) remain constant regardless of the prompt topic.
2. **Tokenizer Log-Probabilities / Perplexity**:
   - For open-weight or leaked tokenizer architectures (Qwen, GLM, Llama), calculating per-token log-probabilities directly reveals token vocabulary preferences.
3. **Synthetic Data Augmentation**:
   - Collect or generate 5,000–10,000 extra prompts and query the 18 frontier APIs (OpenAI, Gemini, Claude, Qwen, GLM, Kimi). With 2,699 samples in the official train set, expanding the training corpus offers the single largest potential leap on the private leaderboard.
4. **Hierarchical Modeling / Stacking**:
   - Level 1: Classify AI Lab / Provider (6 classes: OpenAI, Google, Anthropic, Qwen, ZAI, Moonshot).
   - Level 2: Classify intra-family model (e.g., distinguishing `gpt-5.6-luna` vs `gpt-5.6-sol` vs `gpt-6-astra`).
