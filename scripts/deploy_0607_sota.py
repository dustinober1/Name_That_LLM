"""Script to package and push the 0.607 SOTA ATLAS Ensemble Kernel (Experiment 22).

Integrates all 10 verified datasets and 4 base models:
1. 85% Two-Stage LoRA (Qwen 8B/14B) + Centered Twins + Shared-Prompt Gaussian Head
2. 15% Four-Gaussian Complementary Feature Stack (UltraBERT, Qwen2.5, Claude Specialists)
3. 4-View Multi-Gram LinearSVC on 41k public pool
4. 10% Joseph Adamski's ATLAS 8B sequence likelihood injection
5. 100-Iteration Sinkhorn uniform column mass balancing
"""

import json
import os
import pathlib
import shutil

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_NOTEBOOK = PROJECT_ROOT / "scratch" / "teach_the_model_10_atlas" / "teach-the-model-10-atlas.ipynb"
DEST_DIR = PROJECT_ROOT / "kernels" / "sota_0607"


def prepare_bundle():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    dest_nb = DEST_DIR / "kaggle_submission.ipynb"
    shutil.copyfile(SOURCE_NOTEBOOK, dest_nb)

    # Copy to notebooks/ for versioning
    shutil.copyfile(SOURCE_NOTEBOOK, PROJECT_ROOT / "notebooks" / "kaggle_sota_0607_atlas_ensemble.ipynb")

    metadata = {
        "id": "dustinober/name-that-llm-sota-0607-atlas-ensemble",
        "title": "Name That LLM - SOTA 0.607 ATLAS Ensemble",
        "code_file": "kaggle_submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "dataset_sources": [
            "seyitkaangunes/ntl-extra-api-passages",
            "abdullah1afify/ntl-generations",
            "dophuchung/ntl-proxy-generations",
            "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head",
            "josephadamski91/ntl-atlas-qwen3-8b-v1",
            "lucasboesen/ntl-free-proxy-generations",
            "nishantdahal/ntl-two-stage-lora",
            "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
            "bbucxi/qwen35-9b-safetensors",
            "pjmathematician/ultrabert-380m-ntl",
        ],
        "competition_sources": [
            "name-that-llm",
        ],
        "kernel_sources": [],
        "model_sources": [
            "qwen-lm/qwen-3/Transformers/8b-base/1",
            "qwen-lm/qwen-3/Transformers/14b-base/1",
            "qwen-lm/qwen-3/Transformers/4b-base/1",
            "qwen-lm/qwen2.5/Transformers/7b/1",
        ],
    }

    meta_path = DEST_DIR / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ SOTA 0.607 bundle prepared in: {DEST_DIR}")
    print(f"✓ Notebook: {dest_nb}")
    print(f"✓ Metadata: {meta_path}")


if __name__ == "__main__":
    prepare_bundle()
