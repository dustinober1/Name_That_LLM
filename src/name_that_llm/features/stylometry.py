"""Stylometric and structural feature extraction for frontier LLMs.

Frontier models exhibit distinct linguistic, structural, and typographic signatures
(e.g., Anthropic's nuanced hedging, OpenAI's formatting structure, Google's direct explanations,
Qwen's technical cadence, etc.). These features are invariant to prompt topic.
"""

import re
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Common discourse markers and model tendencies
TRANSITION_MARKERS = [
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\bhowever\b",
    r"\bnevertheless\b",
    r"\bconsequently\b",
    r"\btherefore\b",
    r"\bin addition\b",
    r"\bin summary\b",
    r"\bin conclusion\b",
    r"\bcrucially\b",
    r"\bessentially\b",
    r"\bnotably\b",
    r"\bspecifically\b",
    r"\bultimately\b",
    r"\bfirstly\b",
    r"\bsecondly\b",
    r"\bfinally\b",
]

HEDGING_MARKERS = [
    r"\blikely\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
    r"\barguably\b",
    r"\bsuggests?\b",
    r"\bseems?\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bpotential(ly)?\b",
    r"\btypically\b",
    r"\bgenerally\b",
]

DEFINITIVE_MARKERS = [
    r"\bclearly\b",
    r"\bdefinitely\b",
    r"\bundoubtedly\b",
    r"\balways\b",
    r"\bnever\b",
    r"\bcertainly\b",
    r"\binevitably\b",
    r"\bprecisely\b",
    r"\bexact(ly)?\b",
]

CONTRACTION_PATTERNS = [
    r"\b(it's|it’s)\b",
    r"\b(don't|don’t)\b",
    r"\b(can't|can’t)\b",
    r"\b(won't|won’t)\b",
    r"\b(there's|there’s)\b",
    r"\b(i've|i’ve)\b",
    r"\b(we're|we’re)\b",
    r"\b(they're|they’re)\b",
    r"\b(you're|you’re)\b",
]


class StylometricExtractor(BaseEstimator, TransformerMixin):
    """Extracts a vector of hand-crafted stylometric features for each text passage."""

    def __init__(self) -> None:
        self.feature_names_: List[str] = []

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        """Transform a series or list of text strings into feature matrix."""
        if isinstance(X, pd.Series):
            texts = X.tolist()
        elif isinstance(X, np.ndarray):
            texts = X.tolist()
        else:
            texts = list(X)

        feature_rows = []
        for text in texts:
            feature_rows.append(self._extract_features_single(str(text)))

        df_feat = pd.DataFrame(feature_rows)
        self.feature_names_ = list(df_feat.columns)
        return df_feat.values

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_

    def _extract_features_single(self, text: str) -> dict:
        feats = {}
        char_count = len(text)
        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        sentence_count = max(len(sentences), 1)
        safe_words = max(word_count, 1)
        safe_chars = max(char_count, 1)

        # 1. Length & Basic counts
        feats["char_count"] = char_count
        feats["word_count"] = word_count
        feats["sentence_count"] = sentence_count
        feats["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0.0
        feats["avg_sentence_len_words"] = word_count / sentence_count
        feats["avg_sentence_len_chars"] = char_count / sentence_count
        sentence_lengths = [len(s.split()) for s in sentences]
        feats["std_sentence_length"] = float(np.std(sentence_lengths)) if sentence_lengths else 0.0

        # 2. Lexical richness
        unique_words = set(w.lower() for w in words)
        feats["ttr_lexical_diversity"] = len(unique_words) / safe_words
        word_freqs = pd.Series([w.lower() for w in words]).value_counts() if words else pd.Series(dtype=int)
        hapax_count = (word_freqs == 1).sum() if not word_freqs.empty else 0
        feats["hapax_legomena_ratio"] = hapax_count / safe_words

        # 3. Punctuation densities (normalized per 1,000 characters)
        feats["rate_semicolon"] = (text.count(";") / safe_chars) * 1000
        feats["rate_colon"] = (text.count(":") / safe_chars) * 1000
        feats["rate_comma"] = (text.count(",") / safe_chars) * 1000
        feats["rate_dash"] = (text.count("-") / safe_chars) * 1000
        feats["rate_em_dash"] = ((text.count("—") + text.count("–")) / safe_chars) * 1000
        feats["rate_exclamation"] = (text.count("!") / safe_chars) * 1000
        feats["rate_question"] = (text.count("?") / safe_chars) * 1000
        feats["rate_quotes"] = (
            (text.count('"') + text.count("'") + text.count("“") + text.count("”") + text.count("‘") + text.count("’"))
            / safe_chars
        ) * 1000
        feats["rate_parens"] = ((text.count("(") + text.count(")")) / safe_chars) * 1000
        feats["rate_brackets"] = ((text.count("[") + text.count("]")) / safe_chars) * 1000
        feats["rate_ellipsis"] = ((text.count("...") + text.count("…")) / safe_chars) * 1000

        # 4. Markdown & Formatting signatures
        feats["count_bullet_points"] = len(re.findall(r"(?:^|\n)\s*[-*•]\s+", text))
        feats["count_numbered_lists"] = len(re.findall(r"(?:^|\n)\s*\d+\.\s+", text))
        feats["count_headers"] = len(re.findall(r"(?:^|\n)\s*#{1,6}\s+", text))
        feats["count_bold_markers"] = len(re.findall(r"\*\*[^*]+\*\*", text))
        feats["count_italic_markers"] = len(re.findall(r"(?<!\*)\*[^*]+\*(?!\*)", text))
        feats["count_inline_code"] = len(re.findall(r"`[^`]+`", text))
        feats["count_code_blocks"] = len(re.findall(r"```", text))
        feats["count_math_symbols"] = len(re.findall(r"(\$|\\\(|\\\[|\\begin)", text))
        feats["count_newlines"] = text.count("\n")
        feats["has_markdown_table"] = 1.0 if "|" in text and "-|-" in text else 0.0

        # 5. Casing and typography
        uppercase_chars = sum(1 for c in text if c.isupper())
        feats["ratio_uppercase"] = uppercase_chars / safe_chars
        feats["ratio_digits"] = sum(1 for c in text if c.isdigit()) / safe_chars
        feats["ratio_titlecase_words"] = (
            sum(1 for w in words if w.istitle()) / safe_words if words else 0.0
        )

        # 6. Linguistic & Discourse Markers
        lower_text = text.lower()
        for idx, pattern in enumerate(TRANSITION_MARKERS):
            feats[f"marker_trans_{idx}"] = len(re.findall(pattern, lower_text))

        feats["sum_hedging_words"] = sum(len(re.findall(p, lower_text)) for p in HEDGING_MARKERS)
        feats["sum_definitive_words"] = sum(len(re.findall(p, lower_text)) for p in DEFINITIVE_MARKERS)
        feats["sum_contractions"] = sum(len(re.findall(p, lower_text)) for p in CONTRACTION_PATTERNS)

        # 7. Pronoun orientation
        first_person = len(re.findall(r"\b(i|me|my|mine|we|us|our|ours)\b", lower_text))
        second_person = len(re.findall(r"\b(you|your|yours)\b", lower_text))
        third_person = len(re.findall(r"\b(he|him|his|she|her|hers|they|them|their|theirs|it|its)\b", lower_text))
        feats["pronoun_first_person"] = first_person / safe_words
        feats["pronoun_second_person"] = second_person / safe_words
        feats["pronoun_third_person"] = third_person / safe_words

        return feats
