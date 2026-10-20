# 🧪 Kaggle Experiment & Submission Log

Tracking cross-validation (CV) vs Public Leaderboard (LB) scores, model iterations, and takeaways for the **[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)** competition.

---

## 📊 Leaderboard & Experiment Summary

| # | Submission Name / Kernel Version | Model Family | Key Features & Architecture | Local CV (Bal Acc) | Public LB (Bal Acc) | Gap (CV - LB) | Status |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :-: |
| **v1** | [Name That LLM - Version 1](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline?scriptVersionId=359528236) | Linear Baseline | TF-IDF (Word 1-2, Char 3-5) + Stylometry + Logistic Regression (`C=2.0`, `balanced`) | **0.3245** | **0.2870** | +0.0375 | ✅ Scored |
| **v2-exp** | Experiment 2 (Standalone GBDT) | LightGBM | 54 Stylometric features + 128 TruncatedSVD dense components (`balanced` weights) | **0.2715** | — | — | 🧪 Tested |
| **v2-blend** | [Name That LLM - Version 2](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline) | Ensemble Blend | 85% Linear Baseline + 15% LightGBM (128 SVD + Stylometry) | **0.3282** | **0.3010** | +0.0272 | ✅ Scored |
| **v3-hier** | [Name That LLM - Version 3](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline) | Two-Stage Hierarchical | Global Base Blend (Linear + LightGBM) + 4 Intra-Lab Specialists (OpenAI, Anthropic, Google, ZAI) + Byte Entropy | **0.3463** | **0.3060** | +0.0403 | ✅ Scored |
| **v3-exp-f1** | Experiment 3 (DeBERTa-v3 Fold 1 Benchmark) | Transformer (`deberta-v3-small`) | 3 epochs, `lr=2e-5`, `max_len=384`, class-weighted cross-entropy on MPS | **0.2189** (Fold 1) | — | — | 🧪 Tested |
| **v3-tuned-f1** | Experiment 3 (Tuned DeBERTa-v3 Fold 1) | Transformer (`deberta-v3-small`) | 4 epochs, diff lr (`3e-5` trunk, `1.5e-4` head), weight decay decoupled | **0.2723** (Fold 1) | — | — | 🧪 Tested |
| **v3-blend-f1** | Experiment 3 (Tri-Model Blend on Fold 1) | Ensemble Blend | 60% Linear + 5% LightGBM + 35% Tuned DeBERTa-v3 | **0.3688** (Fold 1) | — | — | 🧪 Tested |
| **v3-5fold** | Experiment 3 (Full 5-Fold DeBERTa-v3) | Transformer (`deberta-v3-small`) | 4 epochs, diff lr (`3e-5` / `1.5e-4`), full 5-fold StratifiedGroupKFold | **0.2504** | — | — | 🧪 Tested |
| **v3-tri-blend** | Experiment 3 (Global Tri-Model Blend) | Ensemble Blend | 57% Linear + 36% DeBERTa-v3 + 7% LightGBM (Full 2,699 OOF) | **0.3549** | — | — | 🧪 Tested |
| **v4-gbdt-tok** | Experiment 4 (LightGBM with Tokenizer Features) | LightGBM | 54 Stylometric + 33 Tokenizer Fingerprints + 128 SVD (215 features) | **0.2752** | — | — | 🧪 Tested |
| **v4-tri-blend** | Experiment 4 (Tri-Model Blend with Tokenizers) | Ensemble Blend | 48% Linear + 50% DeBERTa-v3 + 2% Upgraded LightGBM | **0.3556** | — | — | 🧪 Tested |
| **v5-hierarchical** | Experiment 5 (Hierarchical Specialists) | Two-Stage Classifier | Global Provider Router + 4 Intra-Lab Specialists (OpenAI, Anthropic, Google, ZAI) | **0.3463** | — | — | 🧪 Tested |
| **v5-master-blend** | Experiment 5 (Master 4-Way Ensemble) | Ensemble Blend | 48% Linear + 50% DeBERTa-v3 + 2% Upgraded LightGBM (69.25% Top-3) | **0.3556** | — | — | 🧪 Tested |
| **v4-gpu-sub** | [Name That LLM - Version 4](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline?scriptVersionId=359737151) | GPU Master Ensemble | 50% Hierarchical Base + 50% 5-Fold DeBERTa-v3 Small on CUDA | **0.3556** | **0.3290** | +0.0266 | ✅ Scored |
| **v5-hungarian-sub** | [Name That LLM - Version 5](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline) | GPU Ensemble + Hungarian Matching | 50% Hierarchical Base + 50% DeBERTa-v3 + TF-IDF Prompt Clustering + Hungarian Bipartite Assignment | **0.4327** | **0.3300** | +0.1027 | ✅ Scored |
| **v6-base-f1** | Experiment 10 (DeBERTa-v3 Base Fold 1) | Transformer (`deberta-v3-base`) | 3 epochs, `batch=4`, `accum=2`, `lr=2e-5`, hidden size 768 (86M params) | **0.2967** (Fold 1) | — | — | 🧪 Tested |
| **v6-base-hung** | Experiment 10 (Base + Hierarchical + Hungarian) | Master Ensemble + Hungarian | 50% DeBERTa-v3 Base + 50% Hierarchical Specialist + Hungarian Matching | **0.5006** (Fold 1) | — | — | 🧪 Tested |
| **v6-base-sub** | [Name That LLM - Version 6](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline) | 86M Base GPU + Hungarian Matching | 50% Hierarchical Base + 50% 5-Fold DeBERTa-v3 Base + Prompt Clustering + Hungarian Assignment | **0.4783** | **0.3380** | +0.1403 | ✅ Scored |
| **v7-tri-sub** | [Name That LLM - Version 7](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline) | Tri-Model GPU Master Ensemble | 35% ModernBERT (5 folds) + 45% DeBERTa Base (5 folds) + 20% Hierarchical + Hungarian | **0.5023** | **0.3820** | +0.1203 | ✅ Scored |
| **v8-tri-opt** | [Name That LLM - Version 8](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline) | Tri-Model GPU Ensemble (50% Hierarchical) | 20% ModernBERT (5 folds) + 30% DeBERTa Base (5 folds) + 50% Hierarchical + Hungarian | **0.5194** | **0.3720** | +0.1474 | ✅ Scored |
| **v9-quad-sub** | [Name That LLM - Version 9](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline) | Quad-Model GPU Master Ensemble (630M params) | 30% ModernBERT-large + 25% ModernBERT-base + 25% DeBERTa-base + 20% Hierarchical + Hungarian | **0.5275** | **0.4010** | +0.1265 | ✅ Scored |
| **v10-style-sub** | [Name That LLM - Version 10](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline) | Quad-Model + StyleGrams (Shape/Function-Word) + Sinkhorn | 30% Large + 25% Base + 25% DeBERTa + 20% StyleGram Specialists + Sinkhorn Balancing + Hungarian | **0.5179** | **0.3970** | +0.1209 | ✅ Scored |
| **v11-poe-sub** | [Name That LLM - Version 11](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline) | 5-Way Product-of-Experts Consensus + Hungarian | 20% Large + 15% Base + 10% DeBERTa + 25% Hierarchical + 30% StyleGrams (Geometric Log-Space) | **0.5582** | **0.3960** | +0.1622 | ✅ Scored |
| **v12-neural-sub** | [Name That LLM - Version 12](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline) | Neural-Heavy Master Consensus (85% Neural) + Hungarian | 40% Large + 25% Base + 20% DeBERTa + 10% Hierarchical + 5% StyleGrams (Geometric Log-Space) | **0.5238** | *In Progress* | — | 🚀 Queued |

---

## 📝 Detailed Experiment Notes

### Experiment 1: TF-IDF + Stylometry 5-Fold Baseline (v1)

- **Date:** 2026-10-18
- **Kernel URL:** [Name That LLM - Solution Pipeline - Version 1 (Script ID: 359528236)](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline?scriptVersionId=359528236)
- **Submission Ref:** `57330382`
- **Public Leaderboard Score:** **`0.287`**
- **Local CV Score:** **`0.3245`** (Standard Accuracy: `0.3246`, Macro F1: `0.3096`)
- **Top-3 Accuracy:** `63.36%`
- **Provider-Level Balanced Accuracy:** `54.52%`

#### 1. Setup & Architecture
- **Validation Scheme:** 5-Fold `StratifiedGroupKFold` grouped on `prompt_id` (150 distinct prompts across 2,699 samples).
- **Features Extracted:**
  - Sublinear Word TF-IDF: n-grams (1, 2), max 25,000 features.
  - Sublinear Char TF-IDF: n-grams (3, 5), max 35,000 features.
  - Sublinear Char Word-Boundary TF-IDF: n-grams (2, 4), max 15,000 features.
  - Hand-crafted Stylometric Vector: sentence length stats, punctuation densities (semicolons, em-dashes, colons, quotes), discourse transition markers, hedging vs definitive language counts, and Markdown structure.
- **Model:** Logistic Regression with `C=2.0`, `class_weight='balanced'`, solver `lbfgs`.
- **Ensembling:** Out-of-fold probability averaging across all 5 fold models.

#### 2. Local Per-Class Recall Breakdown
The model shows stark differences across the 18 frontier models:
- **Strong Recall (> 45%):**
  - `gemini-3.1-pro-preview`: **0.70** (105 / 150)
  - `claude-opus-5`: **0.51** (77 / 150)
  - `qwen3.8-max`: **0.49** (74 / 150)
  - `gpt-6-astra`: **0.49** (73 / 150)
  - `claude-opus-4-6`: **0.47** (70 / 150)
- **Moderate Recall (25% - 45%):**
  - `gpt-5.5`: 0.41, `gemini-3.7-flash`: 0.38, `gemini-3.6-flash`: 0.36, `claude-sonnet-5`: 0.35, `gemini-3.8-flash`: 0.29, `kimi-k3`: 0.25
- **Challenging Intra-Family Classes (< 25%):**
  - `gpt-5.6-sol`: **0.07**, `claude-fable-5`: **0.11**, `glm-5.3`: **0.14**, `gpt-5.6-terra`: **0.18**, `glm-5.3-flash`: **0.20**, `claude-opus-4-8`: **0.20**, `gpt-5.6-luna`: **0.24**

#### 3. Key Insights & Diagnostics
1. **Lab vs Model Distinction:** The model achieves **`54.52%`** balanced accuracy when predicting just the lab (OpenAI, Google, Anthropic, Qwen, ZAI, Moonshot). The biggest confusion is intra-lab: distinguishing between different versions/flavors from the same provider (e.g. `gpt-5.6-sol` vs `gpt-5.6-terra` vs `gpt-5.6-luna`).
2. **CV to LB Correlation:** Local CV was `0.325` and Public LB scored `0.287` (a delta of `0.038`). This delta is consistent with evaluating on unseen prompts and test set distribution variance.
3. **Top-3 Recall is 63.36%:** The true model is in the top 3 predictions nearly two-thirds of the time, suggesting that confidence calibration, ensembling, and hierarchical stacking can yield major score jumps.

---

### Experiment 2: LightGBM GBDT with Stylometry + 128 SVD Dense Components

- **Date:** 2026-10-18
- **Local CV Score (Balanced Accuracy):** **`0.2715`**
  - Fold 1: `0.2704` | Fold 2: `0.3056` | Fold 3: `0.2815` | Fold 4: `0.2389` | Fold 5: `0.2611`
  - Standard Accuracy: `0.2716`
  - Macro F1: `0.2493`
  - Top-3 Accuracy: `58.06%`
  - Provider-Level Balanced Accuracy: `51.92%`
  - Multi-class Log Loss: `2.7849`

#### 1. Setup & Architecture
- **Features Extracted:**
  - 54 Hand-crafted Stylometric Features (sentence length statistics, punctuation densities per 1k chars, transition markers, markdown structures, hedging vs definitive language counts).
  - 128 TruncatedSVD dense components fitted on sublinear word & char TF-IDF feature union.
- **Model:** LightGBM Classifier (`objective='multiclass'`, `class_weight='balanced'`, `n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`).
- **Validation:** 5-Fold `StratifiedGroupKFold` grouped on `prompt_id`.

#### 2. Top Discriminative Features
LightGBM's tree splits heavily favored the hand-crafted stylometric features over individual SVD components:
1. `rate_comma` (importance: 2075)
2. `rate_em_dash` (importance: 1857)
3. `avg_word_length` (importance: 1684)
4. `svd_119` (importance: 1633)
5. `svd_121` (importance: 1513)
6. `avg_sentence_len_chars` (importance: 1431)
7. `rate_colon` (importance: 1347)
8. `std_sentence_length` (importance: 1317)
9. `svd_125` (importance: 1311)
10. `svd_120` (importance: 1300)

#### 3. Key Findings & Ensembling Synergy
1. **Why Standalone GBDT Underperforms Linear Baseline (`0.2715` vs `0.3245`):**
   - High-cardinality text classification relies heavily on fine-grained vocabulary and char n-grams. Compressing 60,000+ sparse TF-IDF features down into 128 SVD components creates a severe information bottleneck for tree models, smoothing over exact token signatures that linear models capture cleanly.
2. **Orthogonal Signal for Blending:**
   - Because LightGBM relies heavily on non-linear stylometric interactions (em-dash frequency, comma cadence, word-length variance), its error distribution is complementary to Logistic Regression.
   - **Ensemble Blend (85% Linear + 15% LightGBM):**
     - Balanced Accuracy increases to **`0.3282`** (+0.0037 over linear baseline alone).
     - Top-3 Accuracy increases to **`64.32%`** (+0.96% over linear baseline).

---

### Experiment 3: Pretrained DeBERTa-v3 Fine-Tuning (Fold 1 Benchmark & Tri-Model Blend)

- **Date:** 2026-10-18
- **Model Backbone:** `microsoft/deberta-v3-small` (44M params)
- **Local Fold 1 CV (Initial 3-Epoch Benchmark):** **`0.2189`** Balanced Accuracy (Loss: 2.0594, Top-3: 52.69%, Provider Bal Acc: 46.12%)
- **Local Fold 1 CV (Tuned 4-Epoch with Differential LR):** **`0.2723`** Balanced Accuracy (Loss: 1.7424, Top-3: 62.34%, Provider Bal Acc: 60.95%)
- **Local Fold 1 Tri-Model Blend (60% Linear + 5% LightGBM + 35% Tuned DeBERTa):** **`0.3688`** Balanced Accuracy (**+0.0427** over Linear baseline `0.3262`, **+0.0406** over v2 blend `0.3282`)

#### 1. Setup & Architecture
- **Hardware Acceleration:** Apple Silicon `mps` device, loaded with `dtype=torch.float32` to prevent fp16 disentangled attention NaN underflow.
- **Tuned Training Parameters:** 4 epochs, `batch_size=8`, `lr=3e-5` (backbone trunk), `head_lr=1.5e-4` (5x multiplier on pooler/classifier), linear warmup (10%), gradient clipping (`1.0`), max sequence length 384 tokens.
- **Loss:** Class-weighted cross-entropy loss inversely proportional to training class frequencies.
- **OOF Output:** Saved to `data/processed/oof_microsoft_deberta-v3-small_fold0.parquet`.
- **Model Checkpoint:** Saved to `models/checkpoints/microsoft_deberta-v3-small_fold0`.

#### 2. Key Diagnostic Discoveries
1. **Convergence and Per-Class Recall Surge:**
   - With differential learning rate (`1.5e-4` on head) and 4 epochs, training loss dropped from `2.059` to `1.742`.
   - Standalone Balanced Accuracy jumped from `0.2189` to **`0.2723`**.
   - Provider-level balanced accuracy climbed to **`60.95%`** (beating linear baseline's `54.52%`).
   - Classes with non-zero recall expanded from 10 to 17 out of 18:
     - `qwen3.8-max`: **0.60** (18/30)
     - `claude-opus-4-6`: **0.63** (19/30)
     - `claude-opus-5`: **0.57** (17/30)
     - `gemini-3.1-pro-preview`: **0.53** (16/30)
     - `kimi-k3`: **0.37** (11/30)
     - `gpt-5.5`: **0.37** (11/30)
     - `glm-5.3-flash`: **0.33** (10/30)
     - `claude-opus-4-8`: **0.33** (10/30)
     - `gemini-3.8-flash`: **0.33** (10/30)
     - `gpt-5.6-sol`: **0.27** (8/30)
     - `claude-sonnet-5`: **0.20** (6/30)
2. **Tri-Model Ensembling Synergy (+0.0427 Jump on Fold 1):**
   - Combining:
     - **Linear Baseline (60%):** Robust global n-gram distribution across the entire vocabulary.
     - **LightGBM (5%):** Stylometric syntax interactions.
     - **Tuned DeBERTa-v3 (35%):** Contextual sequence modeling and provider routing.
   - Pushes Fold 1 Balanced Accuracy to **`0.3688`** (+4.27% over Linear, +4.06% over v2 blend).

---

### Experiment 3b: Full 5-Fold DeBERTa-v3 & Global Tri-Model Blend (v3)

- **Date:** 2026-10-18
- **Standalone 5-Fold DeBERTa-v3 CV:** **`0.2504`** Balanced Accuracy (Loss: 2.0497, Top-3: 61.28%, Provider Bal Acc: 53.87%)
- **Global 5-Fold Tri-Model Blend CV:** **`0.3549`** Balanced Accuracy (**+0.0304** over Linear Baseline `0.3245`, **+0.0267** over v2 blend `0.3282`)
  - **Standard Accuracy:** `0.3549`
  - **Macro F1:** `0.3371`
  - **Top-3 Accuracy:** **`68.80%`** (Nearly 70% of true models are in top 3!)
  - **Multi-class Log Loss:** **`1.8558`** (Substantial drop from linear baseline's `2.4102`)
  - **Provider-Level Balanced Accuracy:** **`59.09%`**

#### 1. Optimal Global Blend Weights (Convex Combination over all 2,699 samples)
- **Linear Baseline (TF-IDF + Stylometry):** **`57.0%`**
- **Fine-Tuned DeBERTa-v3 (Contextual Sequence Transformer):** **`36.0%`**
- **LightGBM (Stylometric Features + 128 SVD):** **`7.0%`**

#### 2. Per-Class Recall Progression (v1 Baseline $\rightarrow$ v3 Tri-Model Blend)
| Target Model | v1 Linear Baseline | v3 Tri-Model Blend | Absolute Delta |
| :--- | :---: | :---: | :---: |
| `gemini-3.1-pro-preview` | 0.70 | **0.74** | +0.04 |
| `claude-opus-4-6` | 0.47 | **0.57** | **+0.10** |
| `qwen3.8-max` | 0.49 | **0.56** | **+0.07** |
| `claude-opus-5` | 0.51 | **0.55** | +0.04 |
| `gpt-6-astra` | 0.49 | **0.53** | +0.04 |
| `gpt-5.5` | 0.41 | **0.41** | 0.00 |
| `claude-sonnet-5` | 0.35 | **0.41** | **+0.06** |
| `gemini-3.7-flash` | 0.38 | **0.39** | +0.01 |
| `gemini-3.8-flash` | 0.29 | **0.37** | **+0.08** |
| `kimi-k3` | 0.25 | **0.31** | **+0.06** |
| `gemini-3.6-flash` | 0.36 | 0.29 | -0.07 |
| `gpt-5.6-luna` | 0.24 | **0.25** | +0.01 |
| `claude-opus-4-8` | 0.20 | **0.25** | **+0.05** |
| `glm-5.3-flash` | 0.20 | **0.20** | 0.00 |
| `gpt-5.6-terra` | 0.18 | **0.19** | +0.01 |
| `glm-5.3` | 0.14 | **0.17** | **+0.03** |
| `claude-fable-5` | 0.01 | **0.13** | **+0.12** |
| `gpt-5.6-sol` | 0.07 | 0.06 | -0.01 |

#### 3. Key Findings
1. **Steady Metric Ascent:** Local 5-fold CV climbed steadily: `0.3245` (v1 Linear) $\rightarrow$ `0.3282` (v2 Blend) $\rightarrow$ **`0.3549`** (v3 Tri-Model Blend).
2. **Top-3 Recall Reaches 68.80%:** The true model is now in the top 3 predictions for more than 2 out of every 3 passages.
3. **Persisted Artifact:** Full 2,699-sample blended OOF matrix saved to `data/processed/oof_tri_blend_v3.parquet`.

---

### Experiment 4: Tokenizer Perplexity & Compression Fingerprinting (v4)

- **Date:** 2026-10-18
- **Feature Extractor Module:** `src/name_that_llm/features/perplexity.py` (`TokenizerFeatureExtractor`)
- **Standalone Upgraded LightGBM CV:** **`0.2752`** Balanced Accuracy (Fold 1: `0.2908`, Fold 2: `0.2963`, Fold 3: `0.2796`, Fold 4: `0.2352`, Fold 5: `0.2741`)
  - Standalone gain: **`+0.0037`** over previous stylometry-only LightGBM (`0.2715`).
- **Global Upgraded Tri-Model Blend CV:** **`0.3556`** Balanced Accuracy (**+0.0311** over Linear Baseline `0.3245`)
  - Standard Accuracy: `0.3557`
  - Macro F1: `0.3371`
  - Top-3 Accuracy: `68.21%`
  - Multi-class Log Loss: `1.8529`
  - Optimal Weights: **48.0% Linear + 50.0% DeBERTa-v3 + 2.0% Upgraded LightGBM**

#### 1. Extracted Tokenizer Features (33 New Dimensions $\rightarrow$ 215 Total)
1. **OpenAI Tiktoken Signals (`cl100k_base` & `o200k_base`):**
   - Exact token counts, characters-per-token, and bytes-per-token.
   - Subword byte length statistics (mean, std, and short token fragmentation ratio).
2. **Alibaba Qwen 2.5 Tokenizer (`Qwen/Qwen2.5-7B`):**
   - Native BPE token counts and characters-per-token compression.
   - Subword character length distribution and fragmentation ratios.
3. **Zhipu AI GLM-4 Tokenizer (`THUDM/glm-4-9b-chat`):**
   - Native GLM token counts and characters-per-token compression.
   - Subword length distributions and fragmentation rates.
4. **Cross-Tokenizer Relative Ratios:**
   - Pairwise ratios across all tokenizers (`qwen/cl100k`, `glm/cl100k`, `o200k/cl100k`, `qwen/glm`, `qwen/o200k`, `glm/o200k`).
5. **Byte-Level Entropy Signatures:**
   - `zlib` and `bz2` compression ratios.

#### 2. Key Diagnostic Discoveries
- **`qwen3.8-max` Recall Surges to 60.0%:**
  - In v1 baseline: `49.3%` (74/150).
  - In v3 blend: `56.0%` (84/150).
  - In v4 blend with tokenizers: **`60.0%` (90/150)** (+10.7% recall over baseline).
- **`glm-5.3` Recall Improves:**
  - Standalone LightGBM recall on `glm-5.3` climbed from `14.0%` to **`19.3%`** (29/150).
- **Ensemble Synergy:**
  - Blending these non-linear tokenizer compression interactions with Linear and DeBERTa boosts global CV to **`0.3556`**.
  - Persisted OOF: `data/processed/oof_tri_blend_v4_tokenizer.parquet`.

---

### Experiment 5: Hierarchical 2-Stage Classifier with Intra-Lab Specialists (v5)

- **Date:** 2026-10-18
- **Architecture Module:** `src/name_that_llm/models/hierarchical.py` (`HierarchicalSpecialistClassifier`)
- **Execution Script:** `scripts/train_hierarchical.py`
- **Standalone Hierarchical Specialist CV:** **`0.3463`** Balanced Accuracy (Fold 1: `0.3520`, Fold 2: `0.3556`, Fold 3: `0.3426`, Fold 4: `0.3630`, Fold 5: `0.3370`)
  - **Multi-class Log Loss:** **`1.8443`** (Lowest log loss of any standalone model family)
  - **Top-3 Accuracy:** **`69.25%`** (New peak across all configurations!)
  - **Provider-Level Balanced Accuracy:** **`59.22%`**
- **Master Multi-Model Ensemble CV:** **`0.3556`** Balanced Accuracy
  - Preserved Top-3 Accuracy: `68.21% - 69.25%`
  - Persisted Artifact: `data/processed/oof_hierarchical_specialist_v5.parquet` and `data/processed/oof_master_ensemble_v5.parquet`.

#### 1. Architecture & Design Rationale
1. **The Anti-Cascade Formula:**
   - Naive sequential routing ($P(m) = P(\text{Router}) \times P(\text{Specialist})$) suffered from catastrophic error cascade (achieving only `0.1556` when Stage 1 made mistakes).
   - Under **Conditional Specialist Refinement**, the global multi-model distribution governs provider confidence, while specialized intra-lab classifiers conditionally sharpen the intra-family distributions:
     $$P_{\text{final}}(m \mid x, \text{Lab}) = (1 - \alpha) P_{\text{global}}(m \mid x, \text{Lab}) + \alpha P_{\text{specialist}}(m \mid x, \text{Lab})$$
2. **Four Specialized Intra-Lab Heads:**
   - **`OpenAIExpert`** (5 classes: `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`): Reached **`38.8%`** intra-family accuracy (vs `26.0%` in global linear baseline).
   - **`AnthropicExpert`** (5 classes: `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`).
   - **`GoogleExpert`** (4 classes: `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`).
   - **`ZAIExpert`** (2 classes: `glm-5.3`, `glm-5.3-flash`).
   - **Singletons (`Qwen`, `Moonshot`)**: Inherit marginal router confidence ($P=1.0$).

---

### Experiment 6: Kaggle Submission Version 3 (Hierarchical 2-Stage Classifier)

- **Date:** 2026-10-18
- **Kernel URL:** [Name That LLM - Solution Pipeline - Version 3](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline)
- **Submission Ref:** `57362144`
- **Public Leaderboard Score:** **`0.3060`** (Progress: `0.2870` (v1) $\rightarrow$ `0.3010` (v2) $\rightarrow$ **`0.3060` (v3)**)
- **Local CV Score:** **`0.3463`** Balanced Accuracy (Top-3 Accuracy: `69.25%`, Log Loss: `1.8443`)
- **Key Upgrades Deployed:**
  1. **Two-Stage Conditional Hierarchical Architecture:** Multi-lab global base predictor (85% Linear + 15% LightGBM) coupled with 4 intra-lab specialist heads (`OpenAI`, `Anthropic`, `Google`, `ZAI`) trained exclusively on homogeneous provider subsets.
  2. **Anti-Cascade Probability Blending:** Scaled specialist intra-family probabilities by marginal global provider confidence, preventing error cascading when the provider is misclassified.
  3. **Byte-Level Entropy Signatures:** Embedded `zlib` and `bz2` subword compression metrics into feature vectors.
  4. **Self-Contained Offline Compliance:** Packaged with zero external network calls, completed 5-fold training and private test inference in 283 seconds on Kaggle container.

---

### Experiment 7: Kaggle Submission Version 4 (5-Fold DeBERTa-v3 GPU Ensemble)

- **Date:** 2026-10-18
- **Kernel URL:** [Name That LLM - Solution Pipeline - Version 4](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline?scriptVersionId=359737151)
- **Dataset Attached:** `dustinober/name-that-llm-deberta-checkpoints` (2.8 GB, 5 fine-tuned DeBERTa-v3 Small fold models)
- **Submission Ref:** `57370723`
- **Public Leaderboard Score:** **`0.3290`** (Progress: `0.2870` (v1) $\rightarrow$ `0.3010` (v2) $\rightarrow$ `0.3060` (v3) $\rightarrow$ **`0.3290` (v4)**)
- **Local CV Score:** **`0.3556`** Balanced Accuracy (Top-3 Accuracy: `68.21%`, Multi-class Log Loss: `1.8529`)
- **Key Upgrades Deployed:**
  1. **CUDA Offline Transformer Inference:** Mounted all 5 fine-tuned `microsoft/deberta-v3-small` fold weights via a dedicated private Kaggle dataset.
  2. **50/50 Master Blend:** Blended 50% Hierarchical 2-Stage Specialist predictions + 50% 5-Fold DeBERTa-v3 sequence model posteriors.
  3. **Zero Network Calls:** Full end-to-end container execution completed in 424 seconds on Kaggle Tesla T4 GPU.

---

### Experiment 8: Prompt Clustering & Hungarian Bipartite Matching Post-Processing

- **Date:** 2026-10-18
- **Module Created:** [`src/name_that_llm/post_processing.py`](file:///Users/dustinober/Projects/Data_Science/Name_That_LLM/src/name_that_llm/post_processing.py)
- **Local CV Balanced Accuracy:** **`0.4327`** (A massive **+0.0771 (+7.71%)** gain over raw argmax `0.3556`!)
  - Theoretical upper bound with ground-truth `prompt_id`: **`0.4375`**
  - Standard Accuracy: **`0.4328`**
  - Macro F1: **`0.4318`**
- **The Core Mathematical Insight:**
  1. **Dataset Invariant:** The test set passages are responses to unseen prompts. Each unseen prompt has at most one response from each of the 18 frontier models (no model ever responds twice to the same prompt).
  2. **Prompt Cluster Recovery:** Responses to the same prompt share lexical and topical content. Applying TF-IDF cosine distance with hierarchical clustering (`average` linkage at distance threshold `0.90`) blindly recovers prompt clusters directly from text without prompt labels.
  3. **Hungarian Bipartite Assignment:** Within each prompt cluster of size $M \le 18$, `scipy.optimize.linear_sum_assignment` maximizes total log-likelihood $\sum_i \log P(x_i = m_i)$ under the strict constraint that no two passages in the cluster receive the same model label.
  4. **Eliminating Majority Starvation:** Previously collapsed intra-family classes instantly recovered:
     - `gpt-5.6-sol` recall: `0.07` $\rightarrow$ **`0.2267`**
     - `claude-fable-5` recall: `0.11` $\rightarrow$ **`0.2215`**
     - `glm-5.3` recall: `0.14` $\rightarrow$ **`0.2333`**
     - `gpt-5.6-terra` recall: `0.18` $\rightarrow$ **`0.3200`**
     - `claude-opus-4-8` recall: `0.08` $\rightarrow$ **`0.4267`**
     - `gemini-3.1-pro-preview` recall: `0.70` $\rightarrow$ **`0.8133`**
     - **Every single class achieved precision, recall, and F1 >= 0.22!**

---

### Experiment 9: Kaggle Submission Version 5 (GPU Ensemble + Hungarian Prompt Matching)

- **Date:** 2026-10-18
- **Kernel URL:** [Name That LLM - Solution Pipeline - Version 5](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline)
- **Dataset Attached:** `dustinober/name-that-llm-deberta-checkpoints` (5 fine-tuned DeBERTa-v3 Small fold models)
- **Submission Ref:** `57371212`
- **Public Leaderboard Score:** **`0.3300`** (Progress: `0.2870` (v1) $\rightarrow$ `0.3010` (v2) $\rightarrow$ `0.3060` (v3) $\rightarrow$ `0.3290` (v4) $\rightarrow$ **`0.3300` (v5)**)
- **Local CV Score:** **`0.4327`** Balanced Accuracy (A **+0.0771 (+7.71%)** gain over v4's `0.3556`)
- **Key Upgrades Deployed:**
  1. **Prompt Clustering:** Reconstructed unseen test prompt groups using TF-IDF cosine distance with hierarchical clustering (`average` linkage at distance threshold `0.90`).
  2. **Hungarian Bipartite Assignment:** Evaluated optimal 1-to-1 matching via `scipy.optimize.linear_sum_assignment` across all candidate models within each prompt, enforcing zero model duplicates per prompt.
  3. **GPU Master Ensemble:** Blended 50% Hierarchical Base + 50% 5-Fold DeBERTa-v3 Small before bipartite decoding.

---

### Experiment 10: Backbone Scaling — DeBERTa-v3 Base Benchmark (Breaks 0.50 CV!)

- **Date:** 2026-10-18
- **Model Backbone:** `microsoft/deberta-v3-base` (86M parameters, hidden size 768, 12 layers)
- **Execution Script:** `scripts/train_transformer.py` (`--accum-steps 2`, `--batch-size 4`, `lr=2e-5`)
- **Fold 1 Standalone Score:** **`0.2967`** Balanced Accuracy (Standard Acc: `0.2968`, Top-3: `66.42%`, Loss: `1.8837`)
  - A decisive **+0.0244 (+2.44%)** improvement over tuned DeBERTa-v3 Small (`0.2723`) in just 3 epochs!
- **Fold 1 Tri-Model Synergy:**
  - 50% DeBERTa-v3 Base + 50% Hierarchical Specialists (Raw Argmax): **`0.3836`** (+0.0261 over standalone hierarchical)
  - **★ With Prompt Hungarian Assignment:** **`0.5006` Balanced Accuracy**!
- **Key Takeaways & Path Forward:**
  - Scaling backbone capacity from 44M (Small) to 86M (Base) directly improves class separation on difficult intra-family models (`gpt-5.6-sol` recall reached 30%, `kimi-k3` reached 33%, `gemini-3.7-flash` reached 47%).
  - Blending Base with Hierarchical Specialists and Hungarian Assignment breaks the **`0.50` CV barrier** for the first time.
  - Training all 5 folds of `microsoft/deberta-v3-base` will be our primary engine to push Public Leaderboard toward **`0.45 - 0.55+`**.

---

### Experiment 11: Kaggle Submission Version 6 (86M Base GPU + Hungarian Matching)

- **Date:** 2026-10-19
- **Kernel URL:** [Name That LLM - Solution Pipeline - Version 6](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline)
- **Datasets Attached:** 
  1. `dustinober/name-that-llm-deberta-base-checkpoints` (All 5 fine-tuned `microsoft/deberta-v3-base` 86M fold checkpoints)
  2. `dustinober/name-that-llm-deberta-checkpoints`
- **Submission Ref:** `57375125`
- **Public Leaderboard Score:** **`0.3380`** (Progress: `0.2870` (v1) $\rightarrow$ `0.3010` (v2) $\rightarrow$ `0.3060` (v3) $\rightarrow$ `0.3290` (v4) $\rightarrow$ `0.3300` (v5) $\rightarrow$ **`0.3380` (v6)**)
- **Execution Status:** Scored ✅
- **Local CV Score:** **`0.4783`** Balanced Accuracy (Full 2,699 OOF samples)
  - Standalone DeBERTa-v3 Base 5-Fold: **`0.2486`**
  - Raw 50/50 Master Blend: **`0.3623`**
  - **★ Master Blend + Hungarian Prompt Matching:** **`0.4783`**!
- **Key Upgrades Deployed:**
  1. **86M Backbone Scaling:** Replaced 44M Small model with 86M Base model (12 layers, hidden size 768) trained with class weights and gradient accumulation.
  2. **Multi-Model Synergy:** Blended with Hierarchical Intra-Lab Specialists to maximize precision within lab clusters.
  3. **Bipartite Optimal Transport:** Hungarian matching within prompt clusters strictly prevents duplicate class assignments on unseen prompts.

---

### Experiment 12: ModernBERT-base 5-Fold Training & Tri-Model Synergy

- **Date:** 2026-10-19
- **Model Backbone:** `answerdotai/ModernBERT-base` (22 layers, RoPE rotary embeddings, 8192 context, 50k BPE vocabulary)
- **Training Setup:** 3 epochs, `batch_size=4`, `accum_steps=2`, `lr=3e-5`, `max_length=256` on Apple Silicon MPS (~877 MB memory footprint)
- **Per-Fold Standalone Scores:**
  - Fold 1: **`0.3501`** (Top-3: `69.94%`)
  - Fold 2: **`0.3667`** (Top-3: `72.41%`, Log Loss: `1.8913`)
  - Fold 3: **`0.3685`** (Top-3: `68.89%`, Log Loss: `1.8696`)
  - Fold 4: **`0.3481`**
  - Fold 5: **`0.3519`** (Top-3: `70.74%`)
- **Global 5-Fold Standalone Score:** **`0.3571` Balanced Accuracy** across all 2,699 samples (Top-3: `69.88%`, Provider Acc: `60.27%`). ModernBERT outperforms every other standalone model in the project!
- **★ Global Master Tri-Model Synergy (All 2,699 samples):**
  - 33% ModernBERT + 33% DeBERTa-v3 Base + 34% Hierarchical Specialists (Raw Argmax): **`0.3823`**
  - **★ With Prompt Hungarian Assignment:** **`0.5023` Balanced Accuracy** across the entire 2,699-row dataset!

---

### Experiment 13: Kaggle Submission Version 7 (Tri-Model Master Ensemble on GPU)

- **Date:** 2026-10-19
- **Kernel URL:** [Name That LLM - Solution Pipeline - Version 7](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline)
- **Datasets Attached:**
  1. `dustinober/name-that-llm-modernbert-checkpoints` (All 5 fine-tuned ModernBERT-base fold checkpoints)
  2. `dustinober/name-that-llm-deberta-base-checkpoints` (All 5 fine-tuned DeBERTa-v3 Base fold checkpoints)
  3. `dustinober/name-that-llm-deberta-checkpoints`
- **Submission Ref:** `57377856`
- **Public Leaderboard Score:** **`0.3820`** (Progress: `0.2870` (v1) $\rightarrow$ `0.3010` (v2) $\rightarrow$ `0.3060` (v3) $\rightarrow$ `0.3290` (v4) $\rightarrow$ `0.3300` (v5) $\rightarrow$ `0.3380` (v6) $\rightarrow$ **`0.3820` (v7)**)
- **Execution Status:** Scored ✅
- **Local CV Score:** **`0.5023`** Balanced Accuracy (Full 2,699 OOF samples)
  - Standalone ModernBERT-base 5-Fold: **`0.3571`**
  - Standalone DeBERTa-v3 Base 5-Fold: **`0.2486`**
  - Raw 35/45/20 Tri-Model Blend: **`0.3708`** (33/33/34 Blend: **`0.3823`**)
  - **★ Tri-Model Blend + Hungarian Prompt Matching:** **`0.5023`**!
- **Key Upgrades Deployed:**
  1. **Dual Pretrained Transformer Inference:** Combined ModernBERT-base (22 layers, RoPE, BPE) and DeBERTa-v3 Base (12 layers, disentangled attention, SPM) for maximum neural feature diversity.
  2. **Tri-Model Master Blend:** Blended ModernBERT + DeBERTa Base + 4 Hierarchical Intra-Lab Specialists.
  3. **Optimal Bipartite Matching:** Solves 1-to-1 prompt assignment with Hungarian matching to eliminate duplicates.

---

### Experiment 14: Kaggle Submission Version 8 (50% Hierarchical vs Deep Transformer Generalization)

- **Date:** 2026-10-19
- **Kernel URL:** [Name That LLM - Solution Pipeline - Version 8](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline)
- **Submission Ref:** `57393216`
- **Blend Weights:** `20% ModernBERT + 30% DeBERTa-v3 Base + 50% Hierarchical Specialists` + Hungarian Prompt Matching
- **Local CV Score:** **`0.5194`** (highest CV achieved across full 2,699 samples)
- **Public Leaderboard Score:** **`0.3720`** (vs `0.3820` for v7)
- **Key Takeaway & Diagnostic Insight:**
  - On training prompts (local CV), the shallow Hierarchical Specialist (TF-IDF + Stylometry) achieves very high scores (`0.5194`) by memorizing training topic distributions.
  - However, on the **unseen, out-of-distribution hidden test prompts**, the deep neural representations (`ModernBERT-base` with 22 layers, RoPE, BPE vocabulary + `DeBERTa-v3 Base` with disentangled attention) generalize significantly better.
  - Increasing the shallow model's weight from 20% to 50% reduced generalization on novel prompts.
  - **Rule Established:** The deep transformers must constitute **75%–85% of total ensemble weight**, with the shallow intra-lab specialists reserved as a minor regularizer (15%–20%) to resolve intra-family ties.

---

### Experiment 15: ModernBERT-large 5-Fold Training & Master Quad-Model Synergy

- **Date:** 2026-10-19
- **Model Backbone:** `answerdotai/ModernBERT-large` (395M parameters, 28 layers, 1024 hidden size, 8192 context, RoPE)
- **Training Setup:** 3 epochs, `batch_size=2`, `accum_steps=4`, `lr=1.5e-5`, `max_length=256` on Apple Silicon MPS (`~0.30s/step`)
- **Standalone 5-Fold Scores:**
  - Fold 1: **`0.3427`** (Top-3: `71.80%`, Provider Acc: `61.07%`)
  - Fold 2: **`0.3667`** (Top-3: `73.15%`, Provider Acc: `62.96%`)
  - Fold 3: **`0.3685`** (Top-3: `71.67%`)
  - Fold 4: **`0.3500`**
  - Fold 5: **`0.3481`** (Top-3: `73.33%`, Provider Acc: `62.44%`)
- **Global 5-Fold Standalone Score:** **`0.3567` Balanced Accuracy** across all 2,699 samples
  - **★ Top-3 Accuracy:** **`71.77%`** (highest standalone top-3 accuracy in the project)
  - **★ Provider-Level Accuracy:** **`62.10%`** (highest standalone provider accuracy in the project)
  - Multi-class Log Loss: `1.9261`
- **★ Master Quad-Model Synergy Breakthrough (All 2,699 samples):**
  - ModernBERT-large (395M, 5 folds) + ModernBERT-base (149M, 5 folds) + DeBERTa-v3 Base (86M, 5 folds) + Hierarchical Intra-Lab Specialists
  - **★ Raw Argmax Balanced Accuracy:** **`0.4048`** (breaks past 40% raw accuracy for the first time!)
  - **★ With Prompt Hungarian Assignment:** **`0.5275` Balanced Accuracy** across the entire 2,699-row dataset!

---

### Experiment 16: Version 9 Quad-Model Master Ensemble Scored on Public LB (0.4010)

- **Date:** 2026-10-19
- **Submission Ref:** `57406098`
- **Public Leaderboard Score:** **`0.4010`** (★ Breakthrough: First submission to cross the **0.400** barrier!)
- **Local CV Score:** **`0.5275`** (Balanced Accuracy with Hungarian Matching)
- **Gap (CV - LB):** `+0.1265`
- **Architecture & Checkpoint Composition (630M Parameters Total):**
  - 5-Fold `answerdotai/ModernBERT-large` (395M params, 28 layers, 1024 hidden size) — Weight: `0.30`
  - 5-Fold `answerdotai/ModernBERT-base` (149M params, 22 layers, 768 hidden size) — Weight: `0.25`
  - 5-Fold `microsoft/deberta-v3-base` (86M params, 12 layers, 768 hidden size) — Weight: `0.25`
  - 2-Stage Hierarchical Intra-Lab Specialists (OpenAI, Anthropic, Google, ZAI) — Weight: `0.20`
  - Lexical Unsupervised TF-IDF Prompt Clustering + Hungarian Bipartite Assignment
- **Key Takeaways & Diagnostic Insights:**
  1. **Neural Scaling Confirmed:** Adding `ModernBERT-large` (395M) gave an instant **+0.019 jump** over Version 7 (`0.3820` $\rightarrow$ `0.4010`), confirming that large-capacity transformer trunks capture nuanced stylistic invariants across diverse prompt domains far better than smaller backbones.
  2. **15 Checkpoints Run Under GPU Time Limits:** All 15 transformer fold checkpoints ran within Kaggle's 9-hour execution window on an `NvidiaTeslaT4` without out-of-memory errors using dynamic batching (`batch_size=16`, `max_len=256/384`).
  3. **Robust Bipartite Matching:** The Hungarian algorithm enforces that unseen prompt clusters produce exactly one prediction per passage, unlocking massive metric gains on the hidden test set.

---

### Experiment 17: Version 10 Quad-Model + StyleGrams (Shape/Function-Word) + Sinkhorn Balancing

- **Date:** 2026-10-19
- **Kernel Version:** 10
- **Submission Ref:** `57417097`
- **Public Leaderboard Score:** **`0.3970`**
- **Local Validation (Full 2,699 Samples):** **`0.5179` Balanced Accuracy** (Sinkhorn + Hungarian) vs **`0.5297`** (StyleGrams + Hungarian without Sinkhorn)
- **Status:** ✅ Scored (`0.3970` Public LB)
- **Key Architectural Upgrades Inspired by "The Prompt Is Not the Author":**
  1. **StyleGram Extractor (`src/name_that_llm/features/stylegrams.py`):**
     - **Shape View:** Masks letters to `X` (uppercase) and `x` (lowercase), preserving punctuation, digits, whitespace, markdown formatting, bullet structures, and capitalization rhythms while stripping all prompt topic words.
     - **Function-Word View:** Strips all topical nouns, verbs, and domain terms, replacing them with `'CONTENT'` and keeping ONLY English closed-class grammatical function words (`the`, `and`, `however`, `although`, `which`, `because`).
     - **Stacked Multi-View Extraction:** Stacked raw char (3-5), raw word (1-3, weight 0.7), shape char (2-6, weight 0.5), and function word (1-3, weight 0.5) n-grams.
     - **Standalone Breakthrough:** StyleGrams alone achieves **`0.3675` Balanced Accuracy** across all 2,699 samples (beating raw TF-IDF `0.3245` by **+0.0430**).
  2. **Prompt-Invariant Intra-Lab Specialists:**
     - Replaced standard TF-IDF inside each intra-lab specialist (OpenAI, Anthropic, Google, ZAI) with `StyleGramExtractor`. This prevents intra-family specialists from confusing sibling models based on prompt domain terminology.
  3. **Ablation Insight on Sinkhorn-Knopp Matrix Balancing:**
     - In the competitor's 35k public passage dataset, Sinkhorn balancing was vital because their external pre-training pool was severely class-imbalanced.
     - On the competition test set, where our neural ensemble is already calibrated on balanced data (exactly 150/class), global Sinkhorn matrix balancing slightly over-regularized confident predictions (dropping local CV from `0.5275` to `0.5179`, and Public LB from `0.4010` to `0.3970`).
     - **Key Rule Established:** Avoid global Sinkhorn balancing on our competition-calibrated models; rely on local **Hungarian Bipartite Prompt Matching**, which assigns classes mutually exclusively per prompt cluster without distorting global confidence.


### Experiment 13: 5-Way Product of Experts Consensus Ensemble (v11)

- **Date:** 2026-10-20
- **Kernel Version:** 11
- **Submission Ref:** `57427707`
- **Public Leaderboard Score:** **`0.3960`**
- **Local Validation (Full 2,699 Samples):** **`0.4130` Raw Balanced Accuracy**, **`0.5582` Hungarian Balanced Accuracy** (Mean Fold Score: `0.5583 +/- 0.0189`)
- **Status:** ✅ Scored (`0.3960` Public LB)
- **Diagnostic Takeaways & Root Cause Analysis:**
  1. **Neural vs Linear Generalization on Unseen Prompts:**
     - In **v9** (80% neural weight: 30% ModernBERT-large + 25% ModernBERT-base + 25% DeBERTa-base + 20% Hierarchical), Public LB was **`0.4010`**.
     - In **v11**, we reduced total neural weight from 80% down to 45% and increased TF-IDF / linear models to 55%. Public LB dropped to **`0.3960`**.
     - **Critical Rule:** While linear models (StyleGrams + Stylometry) perform well on local CV, pre-trained transformer backbones (630M params) generalize significantly better to unseen prompt topics on the test set. Future ensembles must maintain $\ge 80\%$ neural weight.
  2. **Competitor Architecture Deconstruction (*"The Prompt Is Not the Author"*):**
     - Competitor notebooks scoring 0.56 - 0.60 rely on two key mechanisms:
       a) **Prompt-Centering / Centered Twins:** Subtracting the prompt mean $\bar{\mathbf{x}}_g$ eliminates prompt topic bias ($\mathbf{x}_i - \bar{\mathbf{x}}_g = \boldsymbol{\mu}_{\text{model}} - \bar{\boldsymbol{\mu}}$). Tested locally on 256 SVD dense vectors: prompt-centering jumped standalone accuracy from `0.3419` to **`0.4149`** (and to **`0.4571`** with Hungarian matching)!
       b) **Public Pretrained LoRA Classifiers:** Top notebooks attach published LoRA checkpoints (`nishantdahal/ntl-two-stage-lora` and `zbigniewkirchner/ntl-qwen3-8b-lora-claude-head`) trained on a 38,642-passage public pool.

---

### Experiment 18: Neural-Heavy Master Consensus Ensemble (v12)

- **Date:** 2026-10-20
- **Kernel Version:** 12
- **Status:** 🚀 Launching on Kaggle GPU
- **Public Leaderboard Score:** *In Progress*
- **Local Validation (Full 2,699 Samples):** **`0.3878` Raw Balanced Accuracy**, **`0.5238` Hungarian Balanced Accuracy**
- **Architecture & Weight Distribution:**
  - 5-Fold `answerdotai/ModernBERT-large` (395M, 28 layers, 1024 hidden) — Weight: `0.40`
  - 5-Fold `answerdotai/ModernBERT-base` (149M, 22 layers, 768 hidden) — Weight: `0.25`
  - 5-Fold `microsoft/deberta-v3-base` (86M, 12 layers, 768 hidden) — Weight: `0.20`
  - StyleGram-Empowered Hierarchical Specialist (OpenAI, Anthropic, Google, ZAI) — Weight: `0.10`
  - StyleGram Linear (Shape & Function-Word Views) — Weight: `0.05`
  - **Total Neural Capacity:** **`85%`** (630M parameters across 15 checkpoints)
  - Geometric Log-Space Consensus Voting (Product of Experts) + Prompt Clustering & Hungarian Assignment
- **Key Hypothesis:**
  - In v9, the 80% neural weighting scored `0.4010` on Public LB.
  - In v11, non-neural weight was increased to 55%, which degraded LB to `0.3960` due to linear models latching onto training prompt domains.
  - v12 restores dominant 85% neural authority while replacing crude TF-IDF with topic-invariant StyleGram representations, targeting **`0.415 - 0.430+`** Public LB.

---

## 🎯 Next Experiment Roadmap

- [x] **Experiment 2 (GBDT & Non-linear Feature Interactions):** Train LightGBM (`scripts/train_lgbm.py`) on stylometric features + 128 SVD dense components.
- [x] **Experiment 3a (Pretrained Transformer Head - Fold 1 Benchmark):** Fine-tune `microsoft/deberta-v3-small` on Fold 1 (`scripts/train_transformer.py`).
- [x] **Experiment 3b (Full 5-Fold DeBERTa-v3 & Tri-Model Blending):** Train all 5 folds of DeBERTa-v3, generate complete OOF matrix, and achieve **0.3549** CV.
- [x] **Experiment 4 (Open-Weight Tokenizer Perplexity Signals):** Extract compression ratios & token counts for Qwen, GLM, and Tiktoken (`src/name_that_llm/features/perplexity.py`), lifting CV to **0.3556**.
- [x] **Experiment 5 (Hierarchical 2-Stage Classifier):** Built `HierarchicalSpecialistClassifier`, achieving **0.3463** standalone CV, **1.8443** log loss, and **69.25%** top-3 accuracy.
- [x] **Kaggle Kernel Submission v3 (Hierarchical Specialist):** Successfully packaged and scored **0.3060** Public LB (`ref: 57362144`).
- [x] **Kaggle Dataset Upload & GPU Master Ensemble (v4):** Uploaded 5-fold DeBERTa-v3 Small checkpoints to Kaggle dataset and scored **0.3290** Public LB (`ref: 57370723`).
- [x] **Experiment 8 (Prompt Clustering & Hungarian Assignment):** Implemented in `src/name_that_llm/post_processing.py`, lifting CV from **0.3556** to **0.4327** (+7.71%).
- [x] **Kaggle Kernel Submission v5 (GPU Ensemble + Hungarian Assignment):** Scored **0.3300** Public LB (`ref: 57371212`).
- [x] **Experiment 10 & 11 (DeBERTa-v3 Base 5-Fold + Kernel Version 6):** Trained all 5 folds of Base (86M), reached **0.4783** CV, uploaded dataset, and launched Version 6!
- [ ] **Phase 7 (Token Attention & LoRA):** Train ModernBERT or Qwen LoRA representations to push beyond **0.55+** LB!





