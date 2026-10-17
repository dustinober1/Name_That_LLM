"""Linear baseline classification models (Logistic Regression, Ridge, SGD) with TF-IDF and Stylometry."""

from typing import Literal, Optional
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from name_that_llm.features.stylometry import StylometricExtractor
from name_that_llm.features.tfidf import build_tfidf_feature_union


class LinearModelPipeline(BaseEstimator, ClassifierMixin):
    """Competitive baseline pipeline combining TF-IDF n-grams + stylometric features with linear classification."""

    def __init__(
        self,
        classifier_type: Literal["logistic", "ridge", "sgd"] = "logistic",
        c: float = 2.0,
        use_stylometry: bool = True,
        max_word_features: int = 25000,
        max_char_features: int = 35000,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ) -> None:
        self.classifier_type = classifier_type
        self.c = c
        self.use_stylometry = use_stylometry
        self.max_word_features = max_word_features
        self.max_char_features = max_char_features
        self.class_weight = class_weight
        self.random_state = random_state

        self.tfidf_ = None
        self.stylometry_ = None
        self.scaler_ = None
        self.clf_ = None
        self.classes_ = None

    def _init_components(self) -> None:
        self.tfidf_ = build_tfidf_feature_union(
            max_word_features=self.max_word_features,
            max_char_features=self.max_char_features,
            include_wb=True,
        )
        if self.use_stylometry:
            self.stylometry_ = StylometricExtractor()
            self.scaler_ = StandardScaler()

        if self.classifier_type == "logistic":
            self.clf_ = LogisticRegression(
                C=self.c,
                max_iter=1500,
                class_weight=self.class_weight,
                random_state=self.random_state,
                solver="lbfgs",
            )
        elif self.classifier_type == "ridge":
            self.clf_ = RidgeClassifier(
                alpha=1.0 / self.c,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        elif self.classifier_type == "sgd":
            self.clf_ = SGDClassifier(
                loss="modified_huber",
                alpha=1.0 / (self.c * 1000),
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=2000,
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def fit(self, texts, y):
        self._init_components()
        X_tfidf = self.tfidf_.fit_transform(texts)

        if self.use_stylometry:
            X_sty = self.stylometry_.fit_transform(texts)
            X_sty_scaled = self.scaler_.fit_transform(X_sty)
            X_all = sp.hstack([X_tfidf, sp.csr_matrix(X_sty_scaled)])
        else:
            X_all = X_tfidf

        self.clf_.fit(X_all, y)
        self.classes_ = self.clf_.classes_
        return self

    def _extract_all_features(self, texts):
        X_tfidf = self.tfidf_.transform(texts)
        if self.use_stylometry:
            X_sty = self.stylometry_.transform(texts)
            X_sty_scaled = self.scaler_.transform(X_sty)
            return sp.hstack([X_tfidf, sp.csr_matrix(X_sty_scaled)])
        return X_tfidf

    def predict(self, texts) -> np.ndarray:
        X_all = self._extract_all_features(texts)
        return self.clf_.predict(X_all)

    def predict_proba(self, texts) -> np.ndarray:
        X_all = self._extract_all_features(texts)
        if hasattr(self.clf_, "predict_proba"):
            return self.clf_.predict_proba(X_all)
        elif hasattr(self.clf_, "decision_function"):
            df = self.clf_.decision_function(X_all)
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            preds = self.predict(texts)
            probs = np.zeros((len(preds), len(self.classes_)))
            for i, p in enumerate(preds):
                idx = np.where(self.classes_ == p)[0][0]
                probs[i, idx] = 1.0
            return probs
