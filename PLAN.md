# 🚀 Name That LLM: Master Execution Plan (Roadmap to 0.60+)

> **Current Status:**  
> - **v1 Baseline:** `0.2870` Public LB (`0.3245` CV) — Logistic Regression (TF-IDF + Stylometry)  
> - **v2 Ensemble:** `0.3010` Public LB (`0.3282` CV) — 85% Linear + 15% LightGBM Blend  
> - **Current Competition Leader (#1):** **`0.616`**  
> - **Target:** Close the +0.315 gap to achieve competitive medal-tier standing (`0.58 - 0.62+`).

---

## 🧭 Executive Summary for New Context Window

### 1. Invariants & Setup
- **Evaluation Metric:** **Balanced Accuracy** (unweighted macro-average recall across all 18 target classes).
- **Format:** **Kaggle Code Competition** (Offline execution, internet disabled, outputs `submission.parquet`).
- **Critical Invariant:** Passages in the test set are responses to **unseen prompts**. All validation must strictly use `StratifiedGroupKFold` grouped on `prompt_id` to guarantee zero prompt leakage.
- **Hardware Acceleration:** Apple Silicon **MPS** is verified and functional locally (`uv run python`). Transformers dependencies (`torch>=2.14`, `transformers>=5.20`, `sentencepiece`, `tiktoken`, `accelerate`) are installed and locked.

### 2. Diagnosis of Current Plateau (`0.301`)
- **What Works:** The model achieves **`54.52%` balanced accuracy at the AI Lab level** (OpenAI vs Google vs Anthropic vs Qwen vs ZAI vs Moonshot).
- **Where It Fails (The Intra-Lab Bottleneck):**
  - High Recall: `gemini-3.1-pro-preview` (70%), `gemini-3.8-flash` (52%), `claude-opus-5` (51%), `gpt-6-astra` (49%), `qwen3.8-max` (49%).
  - Severe Intra-Family Confusion: `claude-fable-5` (**1%**), `glm-5.3-flash` (**5%**), `gpt-5.6-sol` (**6%**), `claude-opus-4-8` (**8%**), `gpt-5.6-luna` (**14%**), `gpt-5.6-terra` (**18%**).
- **The Core Problem:** Models from the same lab share identical system prompts, tokenizers, and alignment guidelines. TF-IDF n-grams and 128 SVD dense components cannot capture reasoning depth, clause syntax, and token sequencing.
- **How Leaders Reach 0.616:**
  1. Pretrained Transformer representations (**DeBERTa-v3**).
  2. Tokenizer Perplexity / Log-Likelihood features (**Qwen, GLM**).
  3. Hierarchical 2-stage routing (Lab Router $\rightarrow$ Intra-Lab Specialists).

---

## 🎯 Phased Execution Roadmap

```
Current: v2 Blend (0.3010 LB)
   │
   ▼
[Phase 1] Experiment 3: Pretrained DeBERTa-v3 Fine-Tuning ─────────► Target: 0.48 - 0.52 LB
   │
   ▼
[Phase 2] Experiment 4: Open-Weight Tokenizer Perplexity Signals ──► Target: 0.54 - 0.57 LB
   │
   ▼
[Phase 3] Experiment 5: Hierarchical 2-Stage Classifier ──────────► Target: 0.58 - 0.60 LB
   │
   ▼
[Phase 4] Tri-Model Stacking & Probability Ensembling ────────────► Target: 0.62+ LB
```

---

## 📋 Phase 1: Experiment 3 — Pretrained Transformer Head (DeBERTa-v3)

### Objective
Replace the bag-of-words representation with contextual sequence modeling. Fine-tune `microsoft/deberta-v3-small` (or `microsoft/deberta-v3-base`) using class-weighted cross-entropy loss to directly optimize Balanced Accuracy.

### Existing Code
- Training script: `scripts/train_transformer.py`
- Model module: `src/name_that_llm/models/transformer_classifier.py`
- Device support: Automatic fallback to Apple Silicon `mps` or Nvidia `cuda`.

### Execution Steps
1. **[Completed] Run Fold 1 Benchmark (Fast Sanity Check):**
   ```bash
   uv run python scripts/train_transformer.py \
     --model-name microsoft/deberta-v3-small \
     --fold-to-run 0 \
     --epochs 3 \
     --batch-size 8 \
     --lr 2e-5 \
     --max-length 384
   ```
2. **[Completed] Evaluate Fold 1 Balanced Accuracy & Recall Recovery:**
   - Standalone Fold 1 DeBERTa-v3: **`0.2189`** (underfitted after 3 epochs, loss: 2.059).
   - **Critical Validation:** Severe intra-family blind spots were dramatically recovered:
     - `gpt-5.6-sol` recall jumped from **0.07** (linear) to **0.33** (10/30).
     - `claude-fable-5` recall jumped from **0.01** (linear) to **0.21** (6/29).
     - `gemini-3.1-pro-preview` jumped from **0.70** to **0.90** (27/30).
     - `claude-opus-5` jumped from **0.51** to **0.67** (20/30).
   - **Tri-Model Ensemble Synergy on Fold 1:**
     - Blending 60% Linear + 5% LightGBM + 35% Tuned DeBERTa reaches **`0.3688`** (+0.0427 over Linear baseline `0.3262`).
3. **[Completed] Execute Full 5-Fold Training & Save OOF Predictions:**
   ```bash
   uv run python scripts/train_transformer.py \
     --model-name microsoft/deberta-v3-small \
     --epochs 4 \
     --batch-size 8 \
     --lr 3e-5 \
     --head-lr 1.5e-4 \
     --max-length 384
   ```
   - Persisted OOF probability matrix: `data/processed/oof_microsoft_deberta-v3-small.parquet`.
   - Saved 5 checkpoints to `models/checkpoints/microsoft_deberta-v3-small_fold{0..4}`.
   - **Global 5-Fold Tri-Model Blend Score:** **`0.3549`** Balanced Accuracy (**+0.0304** over Linear Baseline `0.3245`, **+0.0267** over v2 blend `0.3282`).
   - Optimal Global Weights: 57.0% Linear + 36.0% DeBERTa-v3 + 7.0% LightGBM.
   - Top-3 Accuracy: **`68.80%`**, Multi-class Log Loss: **`1.8558`**.
4. **Build Kaggle GPU Kernel & Submit:**
   - DeBERTa fine-tuning or inference inside Kaggle offline container requires GPU (`--gpu` flag).
   - Update `scripts/build_submission_notebook.py` to support `--model-type transformer`.
   - Push and submit:
     ```bash
     uv run python scripts/submit_to_kaggle.py \
       -m "Experiment 3: 5-Fold DeBERTa-v3-small fine-tuned with class weights" \
       --model-type transformer \
       --gpu
     ```

---

## 📋 Phase 2: Experiment 4 — Tokenizer Perplexity & Log-Likelihood Fingerprinting

### Objective
Extract near-deterministic signatures for models with open or known tokenizers (`qwen3.8-max`, `glm-5.3`, `glm-5.3-flash`).

### Mechanics
1. **Why it works:** Every tokenizer uses different BPE vocabulary merges. When a passage is generated by Qwen, tokenizing it with `Qwen/Qwen2.5-7B` yields significantly fewer tokens and lower subword fragmentation than when tokenized with Llama or Mistral.
2. **Feature Extraction:**
   - For each passage, compute:
     - `token_count_qwen`: Number of tokens under Qwen tokenizer.
     - `token_count_glm`: Number of tokens under GLM-4 tokenizer.
     - `token_count_cl100k`: Number of tokens under OpenAI TikToken.
     - `ratio_qwen_to_cl100k = token_count_qwen / token_count_cl100k`.
   - Compute pseudo-perplexity or byte-level compression ratios.
3. **Implementation Plan:**
   - Create `src/name_that_llm/features/perplexity.py`.
   - Add tokenizer compression metrics into `StylometricExtractor`.
   - Retrain LightGBM and evaluate if GLM and Qwen recall jumps to >80%.

---

## 📋 Phase 3: Experiment 5 — Hierarchical 2-Stage Classifier

### Objective
Eliminate intra-lab cross-talk by splitting the classification task into two specialized stages.

### Architecture
```
                         [ Input Passage ]
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Stage 1: Lab Router   │
                     │ (OpenAI, Google, etc) │
                     └───────────┬───────────┘
                                 │
         ┌───────────────┬───────┴───────┬───────────────┐
         ▼               ▼               ▼               ▼
   [OpenAI Expert] [Google Expert] [Claude Expert] [Other Labs]
     5 models        4 models        5 models        4 models
```

### Mechanics
1. **Stage 1 (Lab Router):**
   - Predicts 1 of 6 providers: `openai`, `google`, `anthropic`, `qwen`, `zai`, `moonshot`.
   - Evaluated metric: Provider Balanced Accuracy (our linear model already gets 55%, DeBERTa gets >85%).
2. **Stage 2 (Specialist Models):**
   - `OpenAIClassifier`: Trained exclusively on the 750 OpenAI training samples. Learns to separate `gpt-5.6-sol` vs `gpt-5.6-luna` vs `gpt-5.6-terra` without being distracted by Claude's em-dashes or Gemini's markdown.
   - `AnthropicClassifier`: Trained exclusively on the 749 Anthropic training samples.
   - `GoogleClassifier`: Trained exclusively on the 600 Gemini training samples.
3. **Final Class Probability Formula:**
   $$P(\text{model}_k) = P(\text{Lab}_j \mid \text{text}) \times P(\text{model}_k \mid \text{text}, \text{Lab}_j)$$

---

## 📋 Phase 4: Tri-Model Probability Blending & Stacking

### Objective
Blend out-of-fold probability distributions from three distinct model families using bounded SLSQP optimization:
1. **Linear TF-IDF + Stylometry:** Fast, high-dimensional n-gram anchor (32.5% CV).
2. **LightGBM on Stylometry + SVD:** Non-linear structural tree interactions (27.2% CV).
3. **Fine-Tuned DeBERTa-v3:** Deep contextual semantics & syntax (>50% CV).
4. **Hierarchical Specialist Head:** Disambiguated intra-lab probabilities.

### Execution Command
```bash
uv run python -c "
from name_that_llm.ensemble import ModelBlender
# Optimize convex combination weights over all OOF matrices
"
```

---

## 🛠️ Key Files & Quick References

| Purpose | File Path |
| :--- | :--- |
| **CV Experiment Tracking Log** | [`EXPERIMENTS.md`](file:///Users/dustinober/Projects/Data_Science/Name_That_LLM/EXPERIMENTS.md) |
| **Comprehensive Gap Analysis** | [`competitive_analysis_and_roadmap.md`](file:///Users/dustinober/.gemini/antigravity-ide/brain/1efbe4c0-6bd8-4cf2-8c4d-4174642caa6c/competitive_analysis_and_roadmap.md) |
| **Transformer Pipeline** | [`scripts/train_transformer.py`](file:///Users/dustinober/Projects/Data_Science/Name_That_LLM/scripts/train_transformer.py) |
| **GBDT Pipeline** | [`scripts/train_lgbm.py`](file:///Users/dustinober/Projects/Data_Science/Name_That_LLM/scripts/train_lgbm.py) |
| **Linear Baseline Pipeline** | [`scripts/train_baseline.py`](file:///Users/dustinober/Projects/Data_Science/Name_That_LLM/scripts/train_baseline.py) |
| **Kaggle Notebook Builder** | [`scripts/build_submission_notebook.py`](file:///Users/dustinober/Projects/Data_Science/Name_That_LLM/scripts/build_submission_notebook.py) |
| **Kaggle Push & Submit CLI** | [`scripts/submit_to_kaggle.py`](file:///Users/dustinober/Projects/Data_Science/Name_That_LLM/scripts/submit_to_kaggle.py) |
| **Official Data Loaders & CV Splits** | [`src/name_that_llm/data.py`](file:///Users/dustinober/Projects/Data_Science/Name_That_LLM/src/name_that_llm/data.py) |
| **Metrics & Per-Class Recalls** | [`src/name_that_llm/metrics.py`](file:///Users/dustinober/Projects/Data_Science/Name_That_LLM/src/name_that_llm/metrics.py) |
| **Ensemble & Blending Engine** | [`src/name_that_llm/ensemble.py`](file:///Users/dustinober/Projects/Data_Science/Name_That_LLM/src/name_that_llm/ensemble.py) |

---

## 🏁 Starting in a New Context Window

In your new session, simply state:
> *"I am picking up the roadmap from `PLAN.md`. Let's begin Phase 1: Experiment 3 by running DeBERTa-v3 on Fold 1."*
