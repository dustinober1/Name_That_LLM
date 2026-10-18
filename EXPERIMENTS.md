# 🧪 Kaggle Experiment & Submission Log

Tracking cross-validation (CV) vs Public Leaderboard (LB) scores, model iterations, and takeaways for the **[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)** competition.

---

## 📊 Leaderboard & Experiment Summary

| # | Submission Name / Kernel Version | Model Family | Key Features & Architecture | Local CV (Bal Acc) | Public LB (Bal Acc) | Gap (CV - LB) | Status |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :-: |
| **v1** | [Name That LLM - Version 1](https://www.kaggle.com/code/dustinober/name-that-llm-solution-pipeline?scriptVersionId=359528236) | Linear Baseline | TF-IDF (Word 1-2, Char 3-5) + Stylometry + Logistic Regression (`C=2.0`, `balanced`) | **0.3245** | **0.2870** | +0.0375 | ✅ Scored |

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

## 🎯 Next Experiment Roadmap

- [ ] **Experiment 2 (GBDT & Non-linear Feature Interactions):** Train LightGBM (`scripts/train_lgbm.py`) on stylometric features + 128 SVD dense components.
- [ ] **Experiment 3 (Pretrained Transformer Head):** Fine-tune `microsoft/deberta-v3-small` using class-weighted loss and 5-fold StratifiedGroupKFold (`scripts/train_transformer.py`).
- [ ] **Experiment 4 (Ensembling & Probability Blending):** Blend out-of-fold probability matrices (Linear + LightGBM + DeBERTa) using SLSQP log-loss optimization (`src/name_that_llm/ensemble.py`).
- [ ] **Experiment 5 (Hierarchical 2-Stage Classifier):**
  - Stage 1: Predict Lab/Provider (OpenAI, Google, Anthropic, Qwen, ZAI, Moonshot).
  - Stage 2: Specialized intra-lab classifiers to disambiguate subtle model differences.
