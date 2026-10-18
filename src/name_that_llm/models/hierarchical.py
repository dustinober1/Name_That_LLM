"""Hierarchical 2-stage classifier with conditional intra-lab specialist heads.

Eliminates intra-lab cross-talk by coupling a global multi-lab probability distribution
with specialized sub-classifiers trained exclusively on individual lab subsets
(OpenAI, Anthropic, Google, ZAI).
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from name_that_llm.config import MODEL_TO_PROVIDER, PROVIDERS, TARGET_MODELS
from name_that_llm.features.perplexity import TokenizerFeatureExtractor
from name_that_llm.features.stylometry import StylometricExtractor


def build_specialist_pipeline(c: float = 1.5, max_iter: int = 400) -> Pipeline:
    """Build a high-dimensional feature extractor + linear classifier for intra-lab disambiguation."""
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                                max_features=15000,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char",
                                ngram_range=(3, 5),
                                sublinear_tf=True,
                                max_features=25000,
                            ),
                        ),
                        (
                            "char_wb_tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(2, 4),
                                sublinear_tf=True,
                                max_features=10000,
                            ),
                        ),
                        ("stylometry", StylometricExtractor()),
                        ("tokenizers", TokenizerFeatureExtractor()),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=max_iter,
                    solver="lbfgs",
                ),
            ),
        ]
    )


class HierarchicalSpecialistClassifier(BaseEstimator, ClassifierMixin):
    """Hierarchical classifier that conditionally sharpens intra-lab probability distributions.

    For any sample, the global distribution determines lab/provider confidence.
    Specialist classifiers trained exclusively on homogeneous provider subsets
    disentangle subtle intra-family model differences without error cascade.
    """

    def __init__(
        self,
        alpha: float = 0.35,
        specialist_c: float = 1.5,
        multi_model_providers: Optional[List[str]] = None,
    ) -> None:
        self.alpha = alpha
        self.specialist_c = specialist_c
        self.multi_model_providers = multi_model_providers or ["OpenAI", "Anthropic", "Google", "ZAI"]
        self.specialists_: Dict[str, Pipeline] = {}
        self.classes_ = np.array(TARGET_MODELS)

    def fit(self, texts: List[str], models: List[str]):
        """Fit intra-lab specialist classifiers on each provider's subset."""
        df = pd.DataFrame({"text": texts, "model": models})
        df["provider"] = df["model"].map(MODEL_TO_PROVIDER)

        for prov in self.multi_model_providers:
            sub = df[df["provider"] == prov]
            if len(sub) == 0:
                continue

            spec = build_specialist_pipeline(c=self.specialist_c)
            spec.fit(sub["text"].values, sub["model"].values)
            self.specialists_[prov] = spec

        return self

    def refine_probabilities(
        self,
        texts: List[str],
        global_probabilities: np.ndarray,
    ) -> np.ndarray:
        """Sharpen a global (N, 18) probability distribution using the intra-lab specialist predictions."""
        n_samples = len(texts)
        refined_probs = global_probabilities.copy()
        text_array = np.array(texts)

        for prov, spec in self.specialists_.items():
            prov_classes = list(spec.classes_)
            prov_col_indices = [TARGET_MODELS.index(m) for m in prov_classes]

            # 1. Total probability assigned to this provider by the global model
            p_prov_global = np.sum(refined_probs[:, prov_col_indices], axis=1, keepdims=True)

            # 2. Intra-lab probability under the global model
            p_global_intra = refined_probs[:, prov_col_indices] / np.maximum(p_prov_global, 1e-12)

            # 3. Specialist intra-lab probability
            p_spec_intra = spec.predict_proba(text_array)

            # 4. Convex combination of global intra-provider and specialist intra-provider
            p_blended_intra = (1.0 - self.alpha) * p_global_intra + self.alpha * p_spec_intra

            # 5. Scale back by marginal provider probability
            refined_probs[:, prov_col_indices] = p_prov_global * p_blended_intra

        # Re-normalize to ensure strictly valid probability distributions
        row_sums = np.sum(refined_probs, axis=1, keepdims=True)
        refined_probs = refined_probs / np.maximum(row_sums, 1e-12)
        return refined_probs

    def predict_from_global(
        self,
        texts: List[str],
        global_probabilities: np.ndarray,
    ) -> List[str]:
        """Predict model labels from refined probabilities."""
        probs = self.refine_probabilities(texts, global_probabilities)
        best_indices = np.argmax(probs, axis=1)
        return [TARGET_MODELS[i] for i in best_indices]
