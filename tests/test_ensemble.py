"""Tests for model blending and probability ensembling."""

import numpy as np
import pytest

from name_that_llm.config import TARGET_MODELS
from name_that_llm.ensemble import ModelBlender


def test_model_blender_weights():
    # 2 classes, 4 samples
    targets = [TARGET_MODELS[0], TARGET_MODELS[1], TARGET_MODELS[0], TARGET_MODELS[1]]

    # Model 1 is completely accurate
    prob1 = np.zeros((4, len(TARGET_MODELS)))
    prob1[0, 0] = 0.99
    prob1[1, 1] = 0.99
    prob1[2, 0] = 0.99
    prob1[3, 1] = 0.99

    # Model 2 is completely inaccurate (reversed)
    prob2 = np.zeros((4, len(TARGET_MODELS)))
    prob2[0, 1] = 0.99
    prob2[1, 0] = 0.99
    prob2[2, 1] = 0.99
    prob2[3, 0] = 0.99

    blender = ModelBlender()
    weights = blender.fit([prob1, prob2], targets)

    # The optimizer should assign virtually all weight to Model 1
    assert weights[0] > 0.8
    assert weights[1] < 0.2

    preds = blender.predict([prob1, prob2])
    assert preds == targets
