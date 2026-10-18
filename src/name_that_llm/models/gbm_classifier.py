"""Gradient Boosting (LightGBM) model pipeline using stylometric, tokenizer, and SVD features."""

from typing import List, Optional
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from name_that_llm.features.perplexity import TokenizerFeatureExtractor
from name_that_llm.features.stylometry import StylometricExtractor
from name_that_llm.features.tfidf import build_tfidf_svd_pipeline


class GBDTClassifierPipeline(BaseEstimator, ClassifierMixin):
    """Pipeline extracting stylometric features, tokenizer fingerprinting, and TF-IDF SVD components, modeled via LightGBM."""

    def __init__(
        self,
        n_svd_components: int = 128,
        use_tokenizer_features: bool = True,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.n_svd_components = n_svd_components
        self.use_tokenizer_features = use_tokenizer_features
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = 1

        self.stylometry_ = StylometricExtractor()
        self.tokenizer_extractor_ = TokenizerFeatureExtractor() if self.use_tokenizer_features else None
        self.svd_pipeline_ = build_tfidf_svd_pipeline(
            n_components=self.n_svd_components, random_state=self.random_state
        )
        self.clf_ = LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
        )
        self.classes_ = None
        self.feature_names_: List[str] = []

    def _extract_features(self, texts, is_fit: bool = False) -> np.ndarray:
        if is_fit:
            feats_sty = self.stylometry_.fit_transform(texts)
            feat_list = [feats_sty]
            names = list(self.stylometry_.get_feature_names_out())

            if self.use_tokenizer_features and self.tokenizer_extractor_ is not None:
                feats_tok = self.tokenizer_extractor_.fit_transform(texts)
                feat_list.append(feats_tok)
                names += list(self.tokenizer_extractor_.get_feature_names_out())

            feats_svd = self.svd_pipeline_.fit_transform(texts)
            feat_list.append(feats_svd)
            names += [f"svd_{i}" for i in range(feats_svd.shape[1])]

            self.feature_names_ = names
            return np.hstack(feat_list)
        else:
            feats_sty = self.stylometry_.transform(texts)
            feat_list = [feats_sty]

            if self.use_tokenizer_features and self.tokenizer_extractor_ is not None:
                feats_tok = self.tokenizer_extractor_.transform(texts)
                feat_list.append(feats_tok)

            feats_svd = self.svd_pipeline_.transform(texts)
            feat_list.append(feats_svd)
            return np.hstack(feat_list)


    def fit(self, texts, y):
        X = self._extract_features(texts, is_fit=True)
        self.clf_.fit(X, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, texts) -> np.ndarray:
        X = self._extract_features(texts, is_fit=False)
        return self.clf_.predict(X)

    def predict_proba(self, texts) -> np.ndarray:
        X = self._extract_features(texts, is_fit=False)
        return self.clf_.predict_proba(X)

    def get_feature_importances(self) -> pd.DataFrame:
        """Return a sorted dataframe of feature importances."""
        importances = self.clf_.feature_importances_
        df = pd.DataFrame(
            {"feature": self.feature_names_, "importance": importances}
        ).sort_values("importance", ascending=False)
        return df.reset_index(drop=True)
