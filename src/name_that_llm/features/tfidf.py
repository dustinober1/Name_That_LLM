"""TF-IDF feature extraction pipelines."""

from typing import Optional
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline


def build_tfidf_feature_union(
    max_word_features: int = 25000,
    max_char_features: int = 35000,
    include_wb: bool = True,
) -> FeatureUnion:
    """Build a multi-level TF-IDF FeatureUnion combining word and character n-grams.

    - Word n-grams (1, 2) capture phrases, common collocations, and vocabulary choices.
    - Character n-grams (3, 5) capture morphemes, punctuation attachments, and sub-word patterns.
    - Char wb n-grams (2, 4) capture word boundary transitions and prefix/suffix tendencies.
    """
    transformers = [
        (
            "word_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=max_word_features,
                sublinear_tf=True,
                min_df=2,
                token_pattern=r"(?u)\b\w+\b",
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=max_char_features,
                sublinear_tf=True,
                min_df=2,
            ),
        ),
    ]

    if include_wb:
        transformers.append(
            (
                "char_wb_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    max_features=15000,
                    sublinear_tf=True,
                    min_df=2,
                ),
            )
        )

    return FeatureUnion(transformers)


def build_tfidf_svd_pipeline(
    n_components: int = 128,
    max_word_features: int = 20000,
    max_char_features: int = 25000,
    random_state: int = 42,
) -> Pipeline:
    """Build a TF-IDF + SVD dimensionality reduction pipeline for dense model inputs (e.g. GBDTs)."""
    return Pipeline(
        [
            (
                "tfidf",
                build_tfidf_feature_union(
                    max_word_features=max_word_features,
                    max_char_features=max_char_features,
                    include_wb=False,
                ),
            ),
            (
                "svd",
                TruncatedSVD(n_components=n_components, random_state=random_state),
            ),
        ]
    )
