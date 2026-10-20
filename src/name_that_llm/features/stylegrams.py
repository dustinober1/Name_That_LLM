"""StyleGrams feature extraction: Shape and Function-Word views for prompt-invariant stylometry."""

import re
from typing import List, Optional
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

FUNCTION_WORDS = frozenset(
    "a an the and or but if while although because as so than that which who whom whose this these those it its "
    "they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with "
    "without for of by over under between through into about is are was were be been being have has had do does did "
    "can could will would shall should may might must not no nor all any some each every both either neither other "
    "another such only also just very too more most less least here there when where why how what whether then now "
    "still yet even much many few little".split()
)
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def shape_view(text: str) -> str:
    """Mask word tokens to 'x' / 'X', preserving capitalization, formatting, punctuation, and markdown syntax."""
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", str(text))


def function_view(text: str) -> str:
    """Replace all topical/content words with 'CONTENT', preserving only closed-class function words and punctuation."""
    return " ".join(
        t.lower() if (t.lower() in FUNCTION_WORDS or not t.isalnum()) else "CONTENT"
        for t in TOKEN_RE.findall(str(text))
    )


class StyleGramExtractor(BaseEstimator, TransformerMixin):
    """Stacked multi-view TF-IDF extractor capturing syntax, formatting, and style without prompt leakage.

    Views:
    1. Raw character 3-5 n-grams: Sub-word suffixes, morphemes, and punctuation attachments.
    2. Raw word 1-3 n-grams: Collocations and stylistic phrasings (weighted 0.7).
    3. Shape character 2-6 n-grams: Capitalization rhythms, list formatting, markdown structure (weighted 0.5).
    4. Function-word 1-3 n-grams: Pure grammatical transitions stripped of prompt topic words (weighted 0.5).
    """

    def __init__(
        self,
        max_char: int = 60000,
        max_word: int = 50000,
        max_shape: int = 40000,
        max_func: int = 40000,
        sublinear_tf: bool = True,
        min_df: int = 2,
    ):
        self.max_char = max_char
        self.max_word = max_word
        self.max_shape = max_shape
        self.max_func = max_func
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df

        self.vec_char = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_features=max_char,
        )
        self.vec_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_features=max_word,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
        )
        self.vec_shape = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 6),
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_features=max_shape,
        )
        self.vec_func = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_features=max_func,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
        )

    def fit(self, texts: List[str], y=None):
        shape_texts = [shape_view(t) for t in texts]
        func_texts = [function_view(t) for t in texts]

        self.vec_char.fit(texts)
        self.vec_word.fit(texts)
        self.vec_shape.fit(shape_texts)
        self.vec_func.fit(func_texts)
        return self

    def transform(self, texts: List[str]) -> csr_matrix:
        shape_texts = [shape_view(t) for t in texts]
        func_texts = [function_view(t) for t in texts]

        X_c = self.vec_char.transform(texts)
        X_w = self.vec_word.transform(texts) * 0.7
        X_s = self.vec_shape.transform(shape_texts) * 0.5
        X_f = self.vec_func.transform(func_texts) * 0.5

        return hstack([X_c, X_w, X_s, X_f], format="csr")
