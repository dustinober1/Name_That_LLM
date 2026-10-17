"""Ensembling, probability calibration, and blending across diverse model families."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from name_that_llm.config import TARGET_MODELS
from name_that_llm.metrics import compute_balanced_accuracy


class ModelBlender:
    """Finds optimal blending weights across model probability predictions."""

    def __init__(self, target_models: Optional[List[str]] = None, metric: str = "log_loss") -> None:
        self.target_models = target_models or TARGET_MODELS
        self.metric = metric
        self.weights_: Optional[np.ndarray] = None

    def fit(self, prob_matrices: List[np.ndarray], y_true: np.ndarray) -> np.ndarray:
        """Find optimal convex combination weights w_i >= 0, sum(w_i) = 1."""
        num_models = len(prob_matrices)
        initial_weights = np.ones(num_models) / num_models

        # If y_true is string labels, convert to indices
        if isinstance(y_true[0], str):
            model_to_idx = {m: i for i, m in enumerate(self.target_models)}
            y_indices = np.array([model_to_idx[y] for y in y_true])
        else:
            y_indices = np.array(y_true)

        if self.metric == "log_loss":
            # Smooth, differentiable multi-class log loss optimization
            def objective(weights):
                w = np.maximum(weights, 1e-6)
                w = w / np.sum(w)
                blended_prob = np.zeros_like(prob_matrices[0])
                for idx, prob in enumerate(prob_matrices):
                    blended_prob += w[idx] * np.clip(prob, 1e-12, 1.0 - 1e-12)
                # Normalize rows
                blended_prob /= np.sum(blended_prob, axis=1, keepdims=True)
                # Compute negative log likelihood
                n_samples = len(y_indices)
                correct_probs = blended_prob[np.arange(n_samples), y_indices]
                return -float(np.mean(np.log(np.maximum(correct_probs, 1e-12))))

            bounds = [(0.0, 1.0) for _ in range(num_models)]
            res = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            )
            weights = np.maximum(res.x, 0)
        else:
            # Direct Balanced Accuracy search via Nelder-Mead
            def objective(weights):
                w = np.maximum(weights, 0)
                if np.sum(w) == 0:
                    return 1.0
                w = w / np.sum(w)
                blended_prob = np.zeros_like(prob_matrices[0])
                for idx, prob in enumerate(prob_matrices):
                    blended_prob += w[idx] * prob
                preds = np.argmax(blended_prob, axis=1)
                return -compute_balanced_accuracy(y_indices, preds)

            res = minimize(
                objective,
                initial_weights,
                method="Nelder-Mead",
            )
            weights = np.maximum(res.x, 0)

        self.weights_ = weights / np.sum(weights)
        return self.weights_

    def blend(self, prob_matrices: List[np.ndarray]) -> np.ndarray:
        """Blend probability predictions using fitted weights."""
        if self.weights_ is None:
            self.weights_ = np.ones(len(prob_matrices)) / len(prob_matrices)

        blended = np.zeros_like(prob_matrices[0])
        for idx, prob in enumerate(prob_matrices):
            blended += self.weights_[idx] * prob
        return blended

    def predict(self, prob_matrices: List[np.ndarray]) -> List[str]:
        """Return target model names corresponding to highest blended probability."""
        blended = self.blend(prob_matrices)
        best_indices = np.argmax(blended, axis=1)
        return [self.target_models[i] for i in best_indices]
