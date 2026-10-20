"""Script to build and package the Two-Stage LoRA submission notebook (0.592 LB tier).

This packages Nishant Dahal's public Two-Stage LoRA pipeline:
1. 4-View Style N-gram LinearSVC trained on train + pool (41k passages)
2. Truncated Qwen3-8B-Base (keep=24 layers) with LoRA adapter + plain & prompt-centered heads
3. Truncated Qwen3-14B-Base (keep=27 layers) with LoRA adapter + plain & prompt-centered heads
4. Sinkhorn-Knopp probability mass balancing across the 18 classes
"""

import json
import os
import pathlib
import sys

KERNEL_DIR = pathlib.Path("kaggle_kernel_lora")
NOTEBOOK_DIR = pathlib.Path("notebooks")


def build_two_stage_lora_notebook(output_path: pathlib.Path = None):
    if output_path is None:
        output_path = NOTEBOOK_DIR / "two_stage_lora_submission.ipynb"

    source_nb_path = pathlib.Path("/tmp/nishant_nb/two-stage-lora.ipynb")
    if not source_nb_path.exists():
        raise FileNotFoundError(f"Source notebook not found at {source_nb_path}")

    with open(source_nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    # Ensure output dirs exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)

    print(f"✓ Saved Two-Stage LoRA submission notebook to: {output_path}")

    # Also prepare kaggle_kernel_lora directory
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    kernel_nb_path = KERNEL_DIR / "kaggle_submission.ipynb"
    with open(kernel_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)

    metadata = {
        "id": "dustinober/name-that-llm-two-stage-lora-pipeline",
        "title": "Name That LLM - Two Stage LoRA Pipeline",
        "code_file": "kaggle_submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "dataset_sources": [
            "nishantdahal/ntl-two-stage-lora"
        ],
        "competition_sources": [
            "name-that-llm"
        ],
        "kernel_sources": [],
        "model_sources": [
            "qwen-lm/qwen-3/Transformers/8b-base/1",
            "qwen-lm/qwen-3/Transformers/14b-base/1"
        ]
    }

    meta_path = KERNEL_DIR / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Prepared Kaggle kernel bundle in: {KERNEL_DIR}")
    print(f"  - Notebook: {kernel_nb_path}")
    print(f"  - Metadata: {meta_path}")


if __name__ == "__main__":
    build_two_stage_lora_notebook()
