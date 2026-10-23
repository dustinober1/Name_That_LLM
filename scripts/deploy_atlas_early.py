"""Script to package and push the Two-Stage LoRA + ATLAS + Early Layers Kernel (Experiment 23 / v17).

Integrates:
1. 80% Two-Stage LoRA (Qwen 8B & 14B) + 4-View Multi-Gram LinearSVC on 41k public pool + Centered Twins
2. 10% Qwen3-4B Early-Layer Syntactic Features (layers 1, 2, 3) + StandardScaler + Logistic Regression
3. 10% ATLAS Qwen3-8B Sequence Likelihoods
4. 100-Iteration Sinkhorn Uniform Column Mass Balancing

Officially verified public score: 0.596 (+0.004 absolute gain over 0.592 baseline).
"""

import json
import os
import pathlib
import shutil

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_NOTEBOOK = PROJECT_ROOT / "scratch" / "two_stage_atlas_early" / "two-stage-lora-atlas-and-early-layers.ipynb"
DEST_DIR = PROJECT_ROOT / "kernels" / "atlas_early"


def prepare_bundle():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    dest_nb = DEST_DIR / "kaggle_submission.ipynb"
    shutil.copyfile(SOURCE_NOTEBOOK, dest_nb)

    # Copy to notebooks/ for repository tracking
    shutil.copyfile(SOURCE_NOTEBOOK, PROJECT_ROOT / "notebooks" / "kaggle_two_stage_atlas_early.ipynb")

    metadata = {
        "id": "dustinober/name-that-llm-two-stage-atlas-early",
        "title": "Name That LLM - Two Stage ATLAS Early",
        "code_file": "kaggle_submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "dataset_sources": [
            "nishantdahal/ntl-two-stage-lora",
            "josephadamski91/ntl-atlas-qwen3-8b-v1",
        ],
        "competition_sources": [
            "name-that-llm",
        ],
        "kernel_sources": [],
        "model_sources": [
            "qwen-lm/qwen-3/Transformers/8b-base/1",
            "qwen-lm/qwen-3/Transformers/14b-base/1",
            "qwen-lm/qwen-3/Transformers/4b-base/1",
        ],
    }

    meta_path = DEST_DIR / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Two-Stage ATLAS Early bundle prepared in: {DEST_DIR}")
    print(f"✓ Notebook: {dest_nb}")
    print(f"✓ Metadata: {meta_path}")


if __name__ == "__main__":
    prepare_bundle()
