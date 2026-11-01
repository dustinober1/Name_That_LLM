"""Build and verify the 5 SOTA Kaggle submissions (v45 - v49).

Portfolio:
- v45 (nlm_v45_open_pool_apex_16): SOTA 0.664 Peak Master (198k Open Pool + svc_w 16.0 + sinkhorn 0.45 + target_mix 0.35 + evidence 1.5x)
- v46 (nlm_v46_atlas_precision_14): Open-Pool High-Precision ATLAS Synergy (198k Open Pool + svc_w 14.0 + atlas_margin 0.28 / atlas_w 0.22 + evidence 1.5x)
- v47 (nlm_v47_sinkhorn_calibrated): Balanced Marginal Horizon (198k Open Pool + svc_w 12.0 + sinkhorn 0.42 + target_mix 0.32 + evidence 1.5x)
- v48 (nlm_v48_pool_weight_tune): 198k Pool Sample Weight Optimization (_srcw 0.45 in SVC + svc_w 12.0 + sinkhorn 0.45 + evidence 1.5x)
- v49 (nlm_v49_grandmaster_champion): Grand Master Champion Consensus (198k Open Pool _srcw 0.40 + svc_w 15.0 + atlas_margin 0.28 / atlas_w 0.22 + sinkhorn 0.45 + target_mix 0.33)
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_DATASETS = [
    "dhruvtibarewal/nlm-l24-adapters",
    "seyitkaangunes/ntl-extra-api-passages",
    "abdullah1afify/ntl-generations",
    "dophuchung/ntl-proxy-generations",
    "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head",
    "dhruvtibarewal/nlm-matched-clean",
    "dhruvtibarewal/nlm-offline-wheels",
    "josephadamski91/ntl-atlas-qwen3-8b-v1",
    "lucasboesen/ntl-free-proxy-generations",
    "nishantdahal/ntl-two-stage-lora",
    "nishantdahal/ntl-handwriting-blend",
    "dhruvtibarewal/nlm-l24-adapters-v5",
    "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
    "pjmathematician/ultrabert-380m-ntl",
]

BASE_KERNEL_SOURCES = [
    "lucasboesen/name-that-llm-glm4-9b-trunc25",
    "lucasboesen/name-that-llm-extract-encoders2",
    "lucasboesen/name-that-llm-electra-large-weights",
    "lucasboesen/name-that-llm-desklib-weights",
    "lucasboesen/name-that-llm-train-lora24-s7",
    "lucasboesen/name-that-llm-train-tokattn",
    "lucasboesen/name-that-llm-train-tokattn2",
    "vnton37nguyn/ntl-train-lora-qwen25-instyle",
]

BASE_MODEL_SOURCES = [
    "alexchilton/llama-3-1-8b-base/Transformers/default/1",
    "qwen-lm/qwen-3/Transformers/14b-base/1",
    "qwen-lm/qwen-3/Transformers/4b-base/1",
    "qwen-lm/qwen-3/Transformers/8b-base/1",
    "qwen-lm/qwen-3/Transformers/14b/1",
    "qwen-lm/qwen-3/Transformers/8b/1",
    "qwen-lm/qwen2.5/Transformers/14b/1",
    "qwen-lm/qwen2.5/Transformers/7b/1",
]


def load_reference_sources():
    nb35 = json.load(open(PROJECT_ROOT / "kernels" / "nlm_v35_sota_0663" / "kaggle_submission.ipynb", encoding="utf-8"))

    setup_src = "".join(nb35["cells"][1]["source"])
    gauss_src = "".join(nb35["cells"][2]["source"])
    infer_src = "".join(nb35["cells"][3]["source"])
    comp_src = "".join(nb35["cells"][4]["source"])
    comp_run_src = "".join(nb35["cells"][5]["source"])

    assert gauss_src.startswith("%%writefile gauss.py")
    assert infer_src.startswith("%%writefile infer.py")
    assert comp_src.startswith("%%writefile comp_stack.py")

    return setup_src, gauss_src, infer_src, comp_src, comp_run_src


def build_open_pool_setup_cell(base_setup_src: str) -> str:
    """Add automatic detection and conversion of crtproduction's 198k open pool."""
    open_pool_block = """
# Open Pool Integration (crtproduction/name-that-llm-open-pool, 198k cleaned passages)
_op_files = glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)
if _op_files:
    _op = pd.read_parquet(_op_files[0], columns=['text', 'model', 'flag_train_prompt_regen'])
    _op[~_op.flag_train_prompt_regen.astype(bool)][['text', 'model']].to_parquet('/kaggle/working/open_pool.parquet')
    POOL = '/kaggle/working/open_pool.parquet'
    print('SVC pool upgraded to 198k Open Pool! Total clean passages:', len(_op[~_op.flag_train_prompt_regen.astype(bool)]))
"""
    return base_setup_src + open_pool_block


def build_main_runner_cell(
    atlas_w: float = 0.20,
    atlas_margin: float = 0.30,
    sinkhorn: float = 0.45,
    sinkhorn_target_mix: float = 0.35,
    svc_w: float = 12.0,
    tag: str = "v45 SOTA 0.664 Peak Master",
) -> str:
    return f"""# Main Inference Execution ({tag})
import json, subprocess, sys, os
import pandas as pd

W_N14, W_N8 = 1.0, 0.5
L = [2, 4, 6, 8, 12, 16, 20, 24]
MODELS = [
  {{"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 3.0, "bs": 8, "max_len": 384, "twin": 0.35, "gauss": 0.70}},
  {{"name": "q8v5", "type": "l24", "path": B8, "adapters": AQ8V5, "keep": 24, "layers": L, "w": 2.0, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.44, "dp": True}},
  {{"name": "ll8v5", "type": "l24", "path": BL, "adapters": ALL8V5, "keep": 24, "layers": L, "w": 1.0, "bs": 16, "max_len": 384, "twin": 0.35, "dp": True}},
  {{"name": "n14b", "type": "l24", "path": B14, "adapters": [N14B], "keep": N14BC['keep'], "layers": N14BC['layers'], "w": 2.5, "bs": 8, "max_len": N14BC['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": N14BC['eos_id']}},
  {{"name": "q25_14", "type": "l24", "path": B25_14, "adapters": [Q25_14], "keep": Q25_14C['keep'], "layers": Q25_14C['layers'], "w": 1.0, "bs": 8, "max_len": Q25_14C['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": Q25_14C['eos_id']}},
  {{"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0}},
]

ATLAS_BUDGET = int(8.3 * 3600 - COMP_SEC - 600)
cmd = [
    sys.executable, 'infer.py', '--data', str(DATA), '--models', json.dumps(MODELS),
    '--perpassage', '1', '--extra', str(EXTRA), '--extra_w', '0.5', '--tfidf_w', '0.25',
    '--distort_w', '0.10', '--svc_w', '{svc_w}', '--pool', str(POOL), '--adapt_w', '1.0',
    '--adapt_thr', '0.15', '--sinkhorn', '{sinkhorn:.2f}', '--sinkhorn_target_mix', '{sinkhorn_target_mix:.2f}', '--post', '0',
    '--atlas_dir', str(ATLAS), '--atlas_base', str(B8), '--atlas_w', '{atlas_w:.2f}', '--atlas_margin', '{atlas_margin:.2f}',
    '--atlas_bs', '4', '--atlas_chunk', '3', '--atlas_budget', str(max(180, ATLAS_BUDGET)),
    '--comp_path', '/kaggle/working/comp_logit.npy', '--comp_w', '0.15',
    '--out', '/kaggle/working/submission.parquet', '--cache', '/kaggle/working/cache'
]

print('RUNNING MAIN INFERENCE ({tag})', flush=True)
if FAST_COMMIT:
    _test_ph = pd.read_csv(os.path.join(DATA, 'test.csv'))
    pd.DataFrame({{'id': _test_ph.id, 'model': 'gpt-5.5'}}).to_parquet('/kaggle/working/submission.parquet', index=False)
else:
    subprocess.run(cmd, check=True)

submission = pd.read_parquet('/kaggle/working/submission.parquet')
assert len(submission) == len(pd.read_csv(os.path.join(DATA, 'test.csv')))
assert set(submission.columns) == {{'id', 'model'}} and submission.id.is_unique and submission.model.notna().all()
print('SUCCESS! Valid submission parquet generated. Rows:', len(submission))
"""


def patch_infer_pool_weight(infer_src: str, pool_srcw: float = 0.45) -> str:
    """Adjust pool aux weight in 4-view SVC from 0.35 to pool_srcw."""
    target = "_P['text'] = _P.text.fillna('').astype(str); _P['_srcw'] = 0.35"
    replacement = f"_P['text'] = _P.text.fillna('').astype(str); _P['_srcw'] = {pool_srcw:.2f}"
    assert target in infer_src, "Failed to find pool _srcw in infer.py"
    return infer_src.replace(target, replacement, 1)


def make_notebook(cells):
    fixed_cells = []
    for i, c in enumerate(cells):
        cell_dict = {
            "cell_type": c["cell_type"],
            "id": f"cell_{i}",
            "metadata": c.get("metadata", {}),
            "source": c["source"] if isinstance(c["source"], list) else [c["source"]],
        }
        if c["cell_type"] == "code":
            cell_dict["execution_count"] = c.get("execution_count", None)
            cell_dict["outputs"] = c.get("outputs", [])
        fixed_cells.append(cell_dict)

    nb = {
        "cells": fixed_cells,
        "metadata": {
            "kaggle": {"accelerator": "gpu", "dataSources": [], "isGpuEnabled": True, "isInternetEnabled": False},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def create_submission_bundle(dir_name, slug, title, cells, extra_datasets=None):
    target_dir = PROJECT_ROOT / "kernels" / dir_name
    target_dir.mkdir(parents=True, exist_ok=True)

    nb = make_notebook(cells)
    with open(target_dir / "kaggle_submission.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    datasets = list(BASE_DATASETS)
    if extra_datasets:
        datasets.extend(extra_datasets)

    meta = {
        "id": slug,
        "title": title,
        "code_file": "kaggle_submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "competition_sources": ["name-that-llm"],
        "dataset_sources": sorted(list(set(datasets))),
        "kernel_sources": sorted(list(set(BASE_KERNEL_SOURCES))),
        "model_sources": sorted(list(set(BASE_MODEL_SOURCES))),
    }
    with open(target_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Successfully generated {dir_name} ({slug})")


def main():
    setup_src, gauss_src, infer_src_base, comp_src, comp_run_src = load_reference_sources()
    open_pool_setup = build_open_pool_setup_cell(setup_src)

    # 1. v45: SOTA 0.664 Peak Master (198k Open Pool + svc_w 16.0 + sinkhorn 0.45 + target_mix 0.35)
    cells_v45 = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Name That LLM — SOTA 0.664 Peak Master (v45)\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [open_pool_setup]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [gauss_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [infer_src_base]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_run_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_main_runner_cell(
            atlas_w=0.20, atlas_margin=0.30, sinkhorn=0.45, sinkhorn_target_mix=0.35, svc_w=16.0, tag="v45 SOTA Peak Master (svc_w 16.0)"
        )]},
    ]
    create_submission_bundle(
        "nlm_v45_open_pool_apex_16",
        "dustinober/name-that-llm-open-pool-apex-16-v45",
        "Name That LLM Open Pool Apex 16 v45",
        cells_v45,
        extra_datasets=["crtproduction/name-that-llm-open-pool"]
    )

    # 2. v46: Open-Pool High-Precision ATLAS Synergy (198k Open Pool + svc_w 14.0 + atlas_margin 0.28 / atlas_w 0.22)
    cells_v46 = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Name That LLM — High-Precision ATLAS Synergy (v46)\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [open_pool_setup]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [gauss_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [infer_src_base]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_run_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_main_runner_cell(
            atlas_w=0.22, atlas_margin=0.28, sinkhorn=0.45, sinkhorn_target_mix=0.35, svc_w=14.0, tag="v46 High-Precision ATLAS (margin 0.28, w 0.22, svc_w 14)"
        )]},
    ]
    create_submission_bundle(
        "nlm_v46_atlas_precision_14",
        "dustinober/name-that-llm-atlas-precision-14-v46",
        "Name That LLM ATLAS Precision 14 v46",
        cells_v46,
        extra_datasets=["crtproduction/name-that-llm-open-pool"]
    )

    # 3. v47: Balanced Marginal Horizon (198k Open Pool + svc_w 12.0 + sinkhorn 0.42 + target_mix 0.32)
    cells_v47 = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Name That LLM — Balanced Marginal Horizon (v47)\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [open_pool_setup]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [gauss_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [infer_src_base]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_run_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_main_runner_cell(
            atlas_w=0.20, atlas_margin=0.30, sinkhorn=0.42, sinkhorn_target_mix=0.32, svc_w=12.0, tag="v47 Marginal Horizon (sinkhorn 0.42, mix 0.32)"
        )]},
    ]
    create_submission_bundle(
        "nlm_v47_sinkhorn_calibrated",
        "dustinober/name-that-llm-sinkhorn-calibrated-v47",
        "Name That LLM Sinkhorn Calibrated v47",
        cells_v47,
        extra_datasets=["crtproduction/name-that-llm-open-pool"]
    )

    # 4. v48: 198k Pool Sample Weight Optimization (_srcw 0.45 in SVC + svc_w 12.0 + sinkhorn 0.45)
    infer_v48 = patch_infer_pool_weight(infer_src_base, pool_srcw=0.45)
    cells_v48 = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Name That LLM — 198k Pool Sample Weight Optimization (v48)\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [open_pool_setup]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [gauss_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [infer_v48]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_run_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_main_runner_cell(
            atlas_w=0.20, atlas_margin=0.30, sinkhorn=0.45, sinkhorn_target_mix=0.35, svc_w=12.0, tag="v48 Pool Sample Weight 0.45"
        )]},
    ]
    create_submission_bundle(
        "nlm_v48_pool_weight_tune",
        "dustinober/name-that-llm-pool-weight-tune-v48",
        "Name That LLM Pool Weight Tune v48",
        cells_v48,
        extra_datasets=["crtproduction/name-that-llm-open-pool"]
    )

    # 5. v49: Grand Master Champion Consensus (_srcw 0.40 + svc_w 15.0 + atlas_margin 0.28 / atlas_w 0.22 + sinkhorn 0.45 + target_mix 0.33)
    infer_v49 = patch_infer_pool_weight(infer_src_base, pool_srcw=0.40)
    cells_v49 = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Name That LLM — Grand Master Champion Consensus (v49)\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [open_pool_setup]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [gauss_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [infer_v49]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_run_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_main_runner_cell(
            atlas_w=0.22, atlas_margin=0.28, sinkhorn=0.45, sinkhorn_target_mix=0.33, svc_w=15.0, tag="v49 Champion Consensus"
        )]},
    ]
    create_submission_bundle(
        "nlm_v49_grandmaster_champion",
        "dustinober/name-that-llm-grandmaster-champion-v49",
        "Name That LLM Grandmaster Champion v49",
        cells_v49,
        extra_datasets=["crtproduction/name-that-llm-open-pool"]
    )

    print("\nAll 5 submission bundles (v45 - v49) generated successfully!")


if __name__ == "__main__":
    main()

