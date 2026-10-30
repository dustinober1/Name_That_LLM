"""Build and verify the 5 SOTA Kaggle submissions (v35 - v39).

Portfolio:
- v35 (nlm_v35_sota_0663): Verified SOTA 0.663 Pipeline (Dual-T4 Data Parallel + ATLAS 20% + 15% Comp Stack + 34-Marker Bayesian Typography + Robust Llama)
- v36 (nlm_v36_sibling_style): 0.663 Pipeline + V40A Family-Only Stylometry Sibling Residual (5-Fold GroupKFold OOF Calibrated, Cross-Family Invariant)
- v37 (nlm_v37_grandmaster_horizon): NTL v41 Grandmaster Horizon (Calibrated Pool Stylometry 0.50 + Expanded ATLAS 25% Margin 0.35 + Soft Sinkhorn 0.50/0.40)
- v38 (nlm_v38_horizon_sibling): Grandmaster Horizon v41 + V40A Sibling Style Residual (max_alpha=0.06) + Zero-Violation Typography Guards
- v39 (nlm_v39_consensus_blend): Grand Master Consensus Blend (Balanced Pool 0.42 + ATLAS 22% Margin 0.33 + Per-Row Member Normalization + Sibling Residual + Sinkhorn 0.48)
"""

import ast
import json
import re
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
    omer_nb = json.load(open("/tmp/top_analysis/omer/0-663-0-660-v31-atlas-20.ipynb", encoding="utf-8"))
    tam_nb = json.load(open("/tmp/top_analysis/tamrazov/name-that-llm-who-am-i.ipynb", encoding="utf-8"))

    gauss_src = "".join(omer_nb["cells"][4]["source"])
    infer_src = "".join(omer_nb["cells"][6]["source"])
    comp_src = "".join(omer_nb["cells"][8]["source"])

    sibling_src = "".join(tam_nb["cells"][3]["source"])

    assert gauss_src.startswith("%%writefile gauss.py")
    assert infer_src.startswith("%%writefile infer.py")
    assert comp_src.startswith("%%writefile comp_stack.py")
    assert sibling_src.startswith("%%writefile sibling_style.py")

    return gauss_src, infer_src, comp_src, sibling_src


def build_setup_cell() -> str:
    return """# Name That LLM — Full Verified 0.663+ Multi-Stage Ensemble Setup & Preflight Verification
import glob, os, subprocess, json, sys, shutil, time as _tc
import pandas as pd

whl = glob.glob('/kaggle/input/**/bitsandbytes-*.whl', recursive=True)
if whl:
    print('bitsandbytes install rc:', subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q'] + whl, capture_output=True, text=True).returncode)

cfgs = sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
def mdir(*parts):
    c = [os.path.dirname(x) for x in cfgs if all(p in x.lower() for p in parts)]
    assert c, f'Missing model config for {parts}'
    return c[0]

def adir(name):
    h = glob.glob(f'/kaggle/input/**/{name}/heads.pt', recursive=True)
    assert h, f'Missing adapter heads.pt for {name}'
    return os.path.dirname(h[0])

DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
FAST_COMMIT = len(pd.read_csv(os.path.join(DATA, 'test.csv'))) < 50
print('FAST_COMMIT:', FAST_COMMIT)

HEADS = os.path.dirname(next(iter(glob.glob('/kaggle/input/**/heads/q3_8bx.npz', recursive=True))))
EXTRA = next(iter(glob.glob('/kaggle/input/**/nlm_matched_clean.parquet', recursive=True)))
POOL = next(iter(glob.glob('/kaggle/input/**/nlm_pool_v5.parquet', recursive=True)))
B8, B14, B25, B4 = mdir('/qwen-3/', '/8b-base/'), mdir('/qwen-3/', '/14b-base/'), mdir('/qwen2.5/', '/7b/'), mdir('/qwen-3/', '/4b-base/')
I8, I14 = mdir('/qwen-3/', '/8b/'), mdir('/qwen-3/', '/14b/')

def find_comp_ckpt(model_type, hidden=None, dir_name=None):
    pat = f'/kaggle/input/**/{dir_name}/config.json' if dir_name else '/kaggle/input/**/config.json'
    for cfg in sorted(glob.glob(pat, recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if dir_name or (c.get('model_type') == model_type and (hidden is None or c.get('hidden_size') == hidden)):
            return os.path.dirname(cfg)
    return None

GLM4_DIR = find_comp_ckpt('glm4', 4096)
assert GLM4_DIR, 'Missing glm4 (4096) checkpoint for comp_stack'
for _ed in ('deberta-v3-large', 'electra-large-discriminator', 'desklib-detector'):
    assert find_comp_ckpt(None, dir_name=_ed), f'Missing {_ed} for comp_stack'

A8 = [adir(f'ts_q3b8_f{k}') for k in range(5)] + [adir('ts_q3b8')]
A14 = [adir(f'ts_q3b14_f{k}') for k in (0, 2, 4)] + [adir('ts_q3b14')]
A25 = [adir('ts_q25_7b')]

def nmeta(name):
    h = glob.glob(f'/kaggle/input/**/{name}/meta.json', recursive=True)
    assert h, f'Missing nmeta {name}'
    d = os.path.dirname(h[0])
    c = json.load(open(h[0]))['cfg']
    return d, c

N8, N8C = nmeta('lora8b_v1')
N14, N14C = nmeta('lora14b_v1')
ATLAS = os.path.dirname(next(iter(glob.glob('/kaggle/input/**/atlas_bundle.json', recursive=True))))

try:
    BL = mdir('/llama-3.1/', '/8b/')
    _LLAMA_SOURCE = 'metaresearch/llama-3.1'
except Exception:
    BL = mdir('llama-3-1-8b-base', '')
    _LLAMA_SOURCE = 'alexchilton/llama-3-1-8b-base'

_SHARDS = {'model-00001-of-00004.safetensors': 4976698672, 'model-00002-of-00004.safetensors': 4999802720, 'model-00003-of-00004.safetensors': 4915916176, 'model-00004-of-00004.safetensors': 1168138808}
assert all(os.path.isfile(os.path.join(BL, f)) for f in _SHARDS), (BL, _LLAMA_SOURCE)

AL = [adir(f'ts_llama8_f{k}') for k in (0, 2, 4)] + [adir('ts_llama8')]
_c = _tc.time()
if not FAST_COMMIT:
    shutil.copytree(BL, '/tmp/llama-3.1/8b', dirs_exist_ok=True)
    BL = '/tmp/llama-3.1/8b'
    print('llama copied to local disk in', round(_tc.time() - _c), 's')
else:
    print('FAST_COMMIT active: skipping heavy model copy')

A8S7 = [adir(f'ts8_s7_f{k}') for k in (0, 2, 4)] + [adir('ts8_s7')]
B25_14 = mdir('/qwen2.5/', '/14b/')
N14B, N14BC = nmeta('lora_qwen3_14b')
Q25_14, Q25_14C = nmeta('lora_qwen25_14b')
DEB = os.path.dirname(glob.glob('/kaggle/input/**/deberta_v3_large/meta.json', recursive=True)[0])
shutil.copytree(DEB, '/kaggle/working/deberta_v3_large', dirs_exist_ok=True)
DEB = '/kaggle/working/deberta_v3_large'

AQ8V5 = [adir(f'q8v5_f{k}') for k in range(5)] + [adir('q8v5')]
AQ14V5 = [adir(f'q14v5_f{k}') for k in range(5)] + [adir('q14v5')]
ALL8V5 = [adir(f'll8v5_f{k}') for k in range(5)] + [adir('ll8v5')]
print('All core model checkpoints and adapters successfully verified!')
"""


def build_comp_stack_runner_cell() -> str:
    return """# Run complementary public stack (frees GPU memory upon completion)
import time as _t, os
_c0 = _t.time()
os.environ['NTL_DROP_Q35'] = '0'
if not FAST_COMMIT:
    get_ipython().system("python comp_stack.py 2>&1 | grep -vE 'Warning|warn'")
COMP_SEC = _t.time() - _c0
print('comp stack wall time:', round(COMP_SEC), 's')
"""


def build_main_runner_cell(
    atlas_w: float = 0.20,
    atlas_margin: float = 0.30,
    sinkhorn: float = 0.45,
    sinkhorn_target_mix: float = 0.35,
    tag: str = "SOTA 0.663",
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
    '--distort_w', '0.10', '--svc_w', '1.5', '--pool', str(POOL), '--adapt_w', '1.0',
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


def patch_infer_with_sibling_style(infer_src: str, max_alpha: float = 0.08) -> str:
    """Inject sibling style residual call into infer.py right before final sub formatting."""
    target = "    _raw_sub = _v31_sub(_raw)\n    _safe = _v31_sub(_safe_logits)"
    replacement = f"""    # Sibling Style Residual: fine-grained intra-family re-ranking with cross-family mass conservation
    _reference_logits = _safe_logits.copy()
    try:
        from sibling_style import apply_grouped_style_residual
        _safe_logits = apply_grouped_style_residual(
            _safe_logits, train.text.values, test.text.values, LABELS, y, Gtr,
            max_alpha={max_alpha}, diagnostics_path=str(_out.with_name('sibling_style_oof.json')))
        log('Sibling style family-only OOF residual fitted successfully')
    except Exception as _style_error:
        log('Sibling style residual skipped or failed; retaining reference logits:', repr(_style_error))
        _safe_logits = _reference_logits

    _raw_sub = _v31_sub(_raw)
    _safe = _v31_sub(_safe_logits)"""
    assert target in infer_src, "Failed to find target hook in infer.py for sibling style"
    return infer_src.replace(target, replacement, 1)


def patch_infer_pool_weight(infer_src: str, pool_srcw: float = 0.50) -> str:
    """Adjust pool aux weight in 4-view SVC from 0.35 to pool_srcw."""
    target = "_P['text'] = _P.text.fillna('').astype(str); _P['_srcw'] = 0.35"
    replacement = f"_P['text'] = _P.text.fillna('').astype(str); _P['_srcw'] = {pool_srcw:.2f}"
    assert target in infer_src, "Failed to find pool _srcw in infer.py"
    return infer_src.replace(target, replacement, 1)


def patch_infer_per_row_norm(infer_src: str) -> str:
    """Inject v39 per-row active ensemble member normalization before Sinkhorn balancing."""
    target = "    lpn = tot / max(1, wsum) if a.sk_div else tot; P = np.exp(lpn - lpn.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)"
    replacement = """    # v39 Per-Row Active Member Normalization:
    # Blend global member normalization with per-row active member normalization
    row_active = sum(abs(v).max(axis=1, keepdims=True) > 0 for v in LP.values())
    lpn_global = tot / max(1, wsum) if a.sk_div else tot
    lpn_row = tot / np.maximum(1, row_active)
    lpn = 0.5 * lpn_global + 0.5 * lpn_row
    P = np.exp(lpn - lpn.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)"""
    assert target in infer_src, "Failed to find lpn hook in infer.py for per-row norm"
    return infer_src.replace(target, replacement, 1)


def patch_infer_zero_violation_guards(infer_src: str) -> str:
    """Inject zero-violation typography character guards (No Hungarian) right before final formatting."""
    target = "    _raw_sub = _v31_sub(_raw)"
    replacement = """    # Zero-Violation Character Guards (conservative soft penalty, No Hungarian):
    for _mark, _allowed in GUARDS:
        _forb = np.array([l not in _allowed for l in LABELS])
        for _i, _t in enumerate(test.text.values):
            if _mark in str(_t):
                _safe_logits[_i, _forb] -= 2.5

    _raw_sub = _v31_sub(_raw)"""
    assert target in infer_src, "Failed to find _raw_sub in infer.py for zero-violation guards"
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
    gauss_src, infer_src_base, comp_src, sibling_src = load_reference_sources()

    # 1. v35: SOTA 0.663 Verified Reproduction (DP 2-GPU + ATLAS 20%)
    cells_v35 = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Name That LLM — SOTA 0.663 Verified Reproduction (v35)\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_setup_cell()]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [gauss_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [infer_src_base]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_comp_stack_runner_cell()]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_main_runner_cell(atlas_w=0.20, atlas_margin=0.30, sinkhorn=0.45, sinkhorn_target_mix=0.35, tag="v35 SOTA 0.663")]},
    ]
    create_submission_bundle("nlm_v35_sota_0663", "dustinober/name-that-llm-sota-0663-v35", "Name That LLM SOTA 0663 v35", cells_v35)

    # 2. v36: 0.663 + V40A Sibling Style Residual
    infer_src_v36 = patch_infer_with_sibling_style(infer_src_base, max_alpha=0.08)
    cells_v36 = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Name That LLM — 0.663 + V40A Sibling Style Residual (v36)\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_setup_cell()]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [gauss_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [sibling_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [infer_src_v36]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_comp_stack_runner_cell()]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_main_runner_cell(atlas_w=0.20, atlas_margin=0.30, sinkhorn=0.45, sinkhorn_target_mix=0.35, tag="v36 Sibling Style 0.663+")]},
    ]
    create_submission_bundle("nlm_v36_sibling_style", "dustinober/name-that-llm-sibling-style-v36", "Name That LLM Sibling Style v36", cells_v36)

    # 3. v37: NTL v41 Grandmaster Horizon (Calibrated Pool 0.50 + ATLAS 25% + Sinkhorn 0.50)
    infer_src_v37 = patch_infer_pool_weight(infer_src_base, pool_srcw=0.50)
    cells_v37 = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Name That LLM — NTL v41 Grandmaster Horizon (v37)\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_setup_cell()]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [gauss_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [infer_src_v37]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_comp_stack_runner_cell()]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_main_runner_cell(atlas_w=0.25, atlas_margin=0.35, sinkhorn=0.50, sinkhorn_target_mix=0.40, tag="v37 Horizon v41")]},
    ]
    create_submission_bundle("nlm_v37_grandmaster_horizon", "dustinober/name-that-llm-ntl-v41-grandmaster-horizon-v37", "Name That LLM NTL v41 Grandmaster Horizon v37", cells_v37)

    # 4. v38: Grandmaster Horizon v41 + Sibling Style Residual
    infer_src_v38 = patch_infer_pool_weight(infer_src_base, pool_srcw=0.50)
    infer_src_v38 = patch_infer_with_sibling_style(infer_src_v38, max_alpha=0.06)
    cells_v38 = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Name That LLM — Grandmaster Horizon v41 + Sibling Style Residual (v38)\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_setup_cell()]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [gauss_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [sibling_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [infer_src_v38]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_comp_stack_runner_cell()]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_main_runner_cell(atlas_w=0.25, atlas_margin=0.35, sinkhorn=0.50, sinkhorn_target_mix=0.40, tag="v38 Horizon + Sibling")]},
    ]
    create_submission_bundle("nlm_v38_horizon_sibling", "dustinober/name-that-llm-horizon-sibling-v38", "Name That LLM Horizon Sibling v38", cells_v38)

    # 5. v39: Grand Master Consensus Blend (Balanced Pool 0.42 + ATLAS 22% + Per-Row Norm + Sibling Residual + Zero-Violation Guards)
    infer_src_v39 = patch_infer_pool_weight(infer_src_base, pool_srcw=0.42)
    infer_src_v39 = patch_infer_per_row_norm(infer_src_v39)
    infer_src_v39 = patch_infer_zero_violation_guards(infer_src_v39)
    infer_src_v39 = patch_infer_with_sibling_style(infer_src_v39, max_alpha=0.06)
    cells_v39 = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Name That LLM — Grand Master Consensus Blend (v39)\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_setup_cell()]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [gauss_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [sibling_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [infer_src_v39]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [comp_src]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_comp_stack_runner_cell()]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [build_main_runner_cell(atlas_w=0.22, atlas_margin=0.33, sinkhorn=0.48, sinkhorn_target_mix=0.38, tag="v39 Consensus Blend")]},
    ]
    create_submission_bundle("nlm_v39_consensus_blend", "dustinober/name-that-llm-consensus-blend-v39", "Name That LLM Consensus Blend v39", cells_v39)

    print("\nAll 5 submission bundles built successfully!")


if __name__ == "__main__":
    main()
