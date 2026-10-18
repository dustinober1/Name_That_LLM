# 🧪 Kaggle Experiment & Submission Log

Tracking cross-validation (CV) vs Public Leaderboard (LB) scores, model iterations, and takeaways for the **[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)** competition.

---

## 📊 Leaderboard & Experiment Summary

| # | Submission Name / Kernel Version | Model Family | Key Features & Architecture | Local CV (Bal Acc) | Public LB (Bal Acc) | Gap (CV - LB) | Status |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :-: |
| **v1** | [Name That LLM - Version 1](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline?scriptVersionId=359528236) | Linear Baseline | TF-IDF (Word 1-2, Char 3-5) + Stylometry + Logistic Regression (`C=2.0`, `balanced`) | **0.3245** | **0.2870** | +0.0375 | ✅ Scored |
| **v2-exp** | Experiment 2 (Standalone GBDT) | LightGBM | 54 Stylometric features + 128 TruncatedSVD dense components (`balanced` weights) | **0.2715** | — | — | 🧪 Tested |
| **v2-blend** | [Name That LLM - Version 2](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline) | Ensemble Blend | 85% Linear Baseline + 15% LightGBM (128 SVD + Stylometry) | **0.3282** | **0.3010** | +0.0272 | ✅ Scored |

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

## 🎯 Next Experiment Roadmap

- [x] **Experiment 2 (GBDT & Non-linear Feature Interactions):** Train LightGBM (`scripts/train_lgbm.py`) on stylometric features + 128 SVD dense components.
- [ ] **Experiment 3 (Pretrained Transformer Head):** Fine-tune `microsoft/deberta-v3-small` using class-weighted loss and 5-fold StratifiedGroupKFold (`scripts/train_transformer.py`).
- [ ] **Experiment 4 (Ensembling & Probability Blending):** Blend out-of-fold probability matrices (Linear + LightGBM + DeBERTa) using SLSQP log-loss optimization (`src/name_that_llm/ensemble.py`).
- [ ] **Experiment 5 (Hierarchical 2-Stage Classifier):**
  - Stage 1: Predict Lab/Provider (OpenAI, Google, Anthropic, Qwen, ZAI, Moonshot).
  - Stage 2: Specialized intra-lab classifiers to disambiguate subtle model differences.
