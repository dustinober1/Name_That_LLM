"""Build and verify the 5 SOTA Kaggle submissions (v30 - v34).

Portfolio:
- v30 (nlm_v27_glm_kimi_v30): Verified 0.654 Pipeline (Full nlm-v27 + Llama-3.1-8B v5 + 15% Comp Stack + 10% ATLAS + v38a 34-Marker Bayesian Typography)
- v31 (nlm_v31_topic_bias): 0.654 Pipeline + v39 Per-Row Active Ensemble Member Normalization (0.5 global + 0.5 row-active before Sinkhorn)
- v32 (nlm_v32_spectral_consensus): Upgraded 120k-Pool Qwen2.5-14B v5 6-Fold Ensemble (dhruvtibarewal/nlm-q25-v5, w=2.0) + 0.654 Typography + Per-Row Norm
- v33 (nlm_v33_sharper_blend): v32 + 3rd Shared-Prompt Gaussian Head (q25_14v5 gauss=0.45) + GLM/Kimi 1,516-Passage Pool Patch + Fixed Soft SVC Topic Bias (LP['svc4'], alpha=0.62)
- v34 (nlm_v34_grand_hybrid): Grand Hybrid (Dual Qwen2.5-14B: q25_14v5 w=2.0 + q25_14 w=0.5) + Pool-Augmented Bayesian Typography Prior (75% Train + 25% 120k Pool) + Zero-Violation Typography Guards (No Hungarian) + Calibrated Sinkhorn (0.85)
"""

import ast
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WHO_AM_I_NB = Path("/tmp/top_kernels/who_am_i/name-that-llm-who-am-i.ipynb")


def load_reference_sources():
    nb = json.load(open(WHO_AM_I_NB, encoding="utf-8"))
    gauss_src = "".join(nb["cells"][2]["source"])
    infer_src = "".join(nb["cells"][3]["source"])
    comp_src = "".join(nb["cells"][4]["source"])
    assert gauss_src.startswith("%%writefile gauss.py")
    assert infer_src.startswith("%%writefile infer.py")
    assert comp_src.startswith("%%writefile comp_stack.py")
    return gauss_src, infer_src, comp_src


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
    "avazaku/name-that-llm-verified-llama-asset-cache",
]

BASE_MODEL_SOURCES = [
    "qwen-lm/qwen-3/Transformers/14b-base/1",
    "qwen-lm/qwen-3/Transformers/4b-base/1",
    "qwen-lm/qwen-3/Transformers/8b-base/1",
    "qwen-lm/qwen-3/Transformers/14b/1",
    "qwen-lm/qwen-3/Transformers/8b/1",
    "qwen-lm/qwen2.5/Transformers/14b/1",
    "qwen-lm/qwen2.5/Transformers/7b/1",
]


def build_setup_cell(require_q25v5: bool = False, require_glm_kimi: bool = False) -> str:
    lines = [
        "# Name That LLM — Full Verified 0.654+ Multi-Stage Ensemble Setup & Preflight Verification\n",
        "import glob, os, subprocess, json, sys, shutil\n",
        "import pandas as pd\n",
        "\n",
        "whl = glob.glob('/kaggle/input/**/bitsandbytes-*.whl', recursive=True)\n",
        "if whl:\n",
        "    print('bitsandbytes install rc:', subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q'] + whl, capture_output=True, text=True).returncode)\n",
        "\n",
        "cfgs = sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))\n",
        "def mdir(*parts):\n",
        "    c = [os.path.dirname(x) for x in cfgs if all(p in x.lower() for p in parts)]\n",
        "    assert c, f'Missing model config for {parts}'\n",
        "    return c[0]\n",
        "\n",
        "def adir(name):\n",
        "    h = glob.glob(f'/kaggle/input/**/{name}/heads.pt', recursive=True)\n",
        "    assert h, f'Missing adapter heads.pt for {name}'\n",
        "    return os.path.dirname(h[0])\n",
        "\n",
        "DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))\n",
        "HEADS = os.path.dirname(next(iter(glob.glob('/kaggle/input/**/heads/q3_8bx.npz', recursive=True))))\n",
        "EXTRA = next(iter(glob.glob('/kaggle/input/**/nlm_matched_clean.parquet', recursive=True)))\n",
        "POOL = next(iter(glob.glob('/kaggle/input/**/nlm_pool_v5.parquet', recursive=True)))\n",
        "B8, B14, B25, B4 = mdir('/qwen-3/', '/8b-base/'), mdir('/qwen-3/', '/14b-base/'), mdir('/qwen2.5/', '/7b/'), mdir('/qwen-3/', '/4b-base/')\n",
        "I8, I14 = mdir('/qwen-3/', '/8b/'), mdir('/qwen-3/', '/14b/')\n",
        "def find_comp_ckpt(model_type, hidden=None, dir_name=None):\n",
        "    pat = f'/kaggle/input/**/{dir_name}/config.json' if dir_name else '/kaggle/input/**/config.json'\n",
        "    for cfg in sorted(glob.glob(pat, recursive=True)):\n",
        "        try:\n",
        "            c = json.load(open(cfg))\n",
        "        except Exception:\n",
        "            continue\n",
        "        if dir_name or (c.get('model_type') == model_type and (hidden is None or c.get('hidden_size') == hidden)):\n",
        "            return os.path.dirname(cfg)\n",
        "    return None\n",
        "\n",
        "GLM4_DIR = find_comp_ckpt('glm4', 4096)\n",
        "assert GLM4_DIR, 'Missing glm4 (4096) checkpoint for comp_stack'\n",
        "for _ed in ('deberta-v3-large', 'electra-large-discriminator', 'desklib-detector'):\n",
        "    assert find_comp_ckpt(None, dir_name=_ed), f'Missing comp_stack encoder {_ed}'\n",
        "\n",
        "def nmeta(name):\n",
        "    h = glob.glob(f'/kaggle/input/**/{name}/meta.json', recursive=True)\n",
        "    assert h, f'Missing meta.json for {name}'\n",
        "    d = os.path.dirname(h[0]); c = json.load(open(h[0]))['cfg']; return d, c\n",
        "\n",
        "ATLAS = os.path.dirname(next(iter(glob.glob('/kaggle/input/**/atlas_bundle.json', recursive=True))))\n",
        "BL = mdir('/llama-3.1/', '/8b/')\n",
        "B25_14 = mdir('/qwen2.5/', '/14b/')\n",
        "N14B, N14BC = nmeta('lora_qwen3_14b')\n",
        "Q25_14, Q25_14C = nmeta('lora_qwen25_14b')\n",
        "DEB_SRC = os.path.dirname(glob.glob('/kaggle/input/**/deberta_v3_large/meta.json', recursive=True)[0])\n",
        "shutil.copytree(DEB_SRC, '/tmp/deberta_v3_large', dirs_exist_ok=True)\n",
        "DEB = '/tmp/deberta_v3_large'\n",
        "\n",
        "AQ8V5 = [adir(f'q8v5_f{k}') for k in range(5)] + [adir('q8v5')]\n",
        "AQ14V5 = [adir(f'q14v5_f{k}') for k in range(5)] + [adir('q14v5')]\n",
        "ALL8V5 = [adir(f'll8v5_f{k}') for k in range(5)] + [adir('ll8v5')]\n",
    ]
    if require_q25v5:
        lines.append("AQ25_14V5 = [adir(f'q25_14v5_f{k}') for k in range(5)] + [adir('q25_14v5')]\n")
        lines.append("print('q25_14v5 adapters verified:', len(AQ25_14V5))\n")
    if require_glm_kimi:
        lines.append("GLM_KIMI_CSV = glob.glob('/kaggle/input/**/ntl_openrouter_glm_kimi.csv', recursive=True)\n")
        lines.append("assert GLM_KIMI_CSV, 'Missing ntl_openrouter_glm_kimi.csv'\n")
        lines.append("print('glm_kimi extra verified:', GLM_KIMI_CSV[0])\n")
    lines.extend([
        "_VISIBLE_TEST = pd.read_csv(os.path.join(DATA, 'test.csv'))\n",
        "_IS_VISIBLE_SAMPLE = len(_VISIBLE_TEST) < 50\n",
        "print('All assets verified! BL=', BL, 'GLM4=', GLM4_DIR, 'IS_VISIBLE_SAMPLE=', _IS_VISIBLE_SAMPLE)\n",
    ])
    return "".join(lines)


def build_infer_py(
    base_infer_src: str,
    mode: str,
) -> str:
    """Customize infer.py based on experiment mode:
    - 'v30': 0.654 pure (global wsum normalization before Sinkhorn + v38a 34-marker Bayesian typography)
    - 'v31': 0.654 + v39 per-row active member normalization (0.5 global + 0.5 row-active)
    - 'v32': same as v31 (used with upgraded q25_14v5 in MODELS)
    - 'v33': v31 + GLM/Kimi pool patch + fixed Soft SVC Topic Bias Attenuation on LP['svc4']
    - 'v34': v31 + Pool-Augmented Bayesian Typography Prior (75% Train + 25% 120k Pool) + Zero-Violation Impossible-Mark Guards (No Hungarian) + Sinkhorn 0.85
    """
    src = base_infer_src

    # 1. For v30, use pure 0.654 (_P_ref only instead of 0.5 _P_ref + 0.5 _P_row)
    if mode == "v30":
        src = src.replace(
            "    tot = 0.5 * _v39_original_pre_post + 0.5 * _v39_normalized_pre_post\n",
            "    tot = _v39_original_pre_post.copy()\n",
        )

    # 2. For v33, add GLM/Kimi pool patch (deduplicated against train/test) + Soft SVC Topic Bias on LP['svc4']
    if mode == "v33":
        old_pool_block = "    _P = pd.read_parquet(a.pool); _P = _P[_P.model.isin(LABELS)]\n"
        new_pool_block = (
            "    _P = pd.read_parquet(a.pool); _P = _P[_P.model.isin(LABELS)]\n"
            "    _gk_files = glob.glob('/kaggle/input/**/ntl_openrouter_glm_kimi.csv', recursive=True)\n"
            "    if _gk_files:\n"
            "        _gk = pd.read_csv(_gk_files[0])[['model', 'text']].dropna()\n"
            "        _gk = _gk[_gk.model.isin(LABELS)]\n"
            "        _forb_tx = set(train.text.astype(str)).union(set(test.text.astype(str)))\n"
            "        _gk = _gk[~_gk.text.astype(str).isin(_forb_tx)].drop_duplicates('text')\n"
            "        _P = pd.concat([_P, _gk], ignore_index=True)\n"
            "        log(f'svc4 pool patched with {len(_gk)} clean glm/kimi rows -> total {len(_P)}')\n"
        )
        assert old_pool_block in src, "Failed to locate _P pool load in infer.py"
        src = src.replace(old_pool_block, new_pool_block, 1)

        # Also add glob import at top if not present
        src = src.replace("import os, sys, time, gc, argparse, json\n", "import os, sys, time, gc, argparse, json, glob\n", 1)

        # Add soft topic bias correction on LP['svc4'] before combining tot
        old_tot_line = "tot = sum(LP.values())\n"
        topic_block = (
            "if 'svc4' in LP and LP['svc4'] is not None:\n"
            "    _svc_raw = np.asarray(LP['svc4'], dtype=np.float64)\n"
            "    _svc_cent = _svc_raw.copy()\n"
            "    _, _ginv, _gcnt = np.unique(Gte, return_inverse=True, return_counts=True)\n"
            "    _elig = (_gcnt[_ginv] >= 8) & (_gcnt[_ginv] <= 18)\n"
            "    for _g in np.unique(Gte[_elig]):\n"
            "        _ix = (Gte == _g)\n"
            "        _svc_cent[_ix] -= _svc_raw[_ix].mean(0, keepdims=True)\n"
            "    def _lsm_np(_z):\n"
            "        _z = _z - _z.max(1, keepdims=True)\n"
            "        return _z - np.log(np.exp(_z).sum(1, keepdims=True))\n"
            "    _delta = (_lsm_np(_svc_cent) - _lsm_np(_svc_raw)) * a.svc_w\n"
            "    _delta[~_elig] = 0.0\n"
            "    _core_tmp = sum(LP.values())\n"
            "    _ord_tmp = np.sort(_core_tmp, axis=1)\n"
            "    _tgate = np.clip((2.0 - (_ord_tmp[:, -1] - _ord_tmp[:, -2])) / 1.6, 0.15, 1.0)\n"
            "    _tgate[~_elig] = 0.0\n"
            "    LP['svc4'] = (LP['svc4'] + 0.62 * _delta * _tgate[:, None]).astype(np.float32)\n"
            "    log(f'applied soft topic bias correction on svc4 for {int(_elig.sum())} dense-cluster rows')\n"
            "tot = sum(LP.values())\n"
        )
        assert old_tot_line in src, "Failed to locate tot = sum(LP.values())"
        src = src.replace(old_tot_line, topic_block, 1)

    # 3. For v34, use Pool-Augmented Bayesian Typography Prior (75% Train + 25% 120k Pool)
    # AND apply the 5 zero-violation impossible-mark guards AFTER Bayesian typography (with NO Hungarian matching!)
    if mode == "v34":
        target_end = (
            "            _pool_sub = _v38_sub(_raw + _gate[:, None] * _ep)\n"
            "            _pool_sub.to_parquet(_out.with_name('candidate_v39_public_prior.parquet'), index=False)\n"
            "            log('v39 optional public prior', len(_pool_df), 'texts')\n"
        )
        replacement_end = (
            "            _hybrid_logits = _raw + _gate[:, None] * _ep\n"
            "            for _mark, _allowed in [(' \\u2014 ', {'claude-opus-4-6', 'claude-opus-5', 'glm-5.3', 'glm-5.3-flash'}),\n"
            "                                    ('\\u2019', {l for l in LABELS if not l.startswith(('claude', 'glm'))}),\n"
            "                                    ('\"', {'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'kimi-k3'}),\n"
            "                                    ('\\u201c', {'gemini-3.8-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max'}),\n"
            "                                    ('\\u2013', {'claude-opus-5', 'gemini-3.1-pro-preview', 'glm-5.3-flash', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'kimi-k3'})]:\n"
            "                _forb = np.array([l not in _allowed for l in LABELS])\n"
            "                for _i, _t in enumerate(test.text.values):\n"
            "                    if _mark in _t and LABELS[int(_hybrid_logits[_i].argmax())] not in _allowed:\n"
            "                        _hybrid_logits[_i, _forb] = -1e9\n"
            "            _safe = _v38_sub(_hybrid_logits)\n"
            "            _safe.to_parquet(_out, index=False)\n"
            "            log('v34 pool-prior typobayes + zero-violation guards written to', _out, 'with', len(_pool_df), 'pool texts')\n"
        )
        assert target_end in src, "Failed to locate v39 optional public prior block"
        src = src.replace(target_end, replacement_end, 1)

    return src


def build_stage_a_cell() -> str:
    return (
        "# Stage A: Complementary Frozen-Feature Stack (runs on hidden competition rerun; fast preflight on 3-row commit)\n"
        "import time as _t, os, numpy as np, pandas as pd\n"
        "\n"
        "# Write initial valid fallback submission immediately\n"
        "pd.DataFrame({'id': _VISIBLE_TEST['id'], 'model': ['gpt-5.5'] * len(_VISIBLE_TEST)}).to_parquet('/kaggle/working/submission.parquet', index=False)\n"
        "\n"
        "if _IS_VISIBLE_SAMPLE:\n"
        "    COMP_SEC = 0\n"
        "    print('Visible 3-row commit sample: skipping 2-hour GPU passes; running preflight syntax & math verification')\n"
        "else:\n"
        "    _c0 = _t.time()\n"
        "    os.environ['NTL_DROP_Q35'] = '1'\n"
        "    !python comp_stack.py 2>&1 | grep -vE 'Warning|warn'\n"
        "    COMP_SEC = _t.time() - _c0\n"
        "    print('comp stack wall time', round(COMP_SEC))\n"
        "    if os.path.exists('/kaggle/working/comp_logit.npy') and os.path.exists('/kaggle/working/comp_meta.json'):\n"
        "        _cm = json.load(open('/kaggle/working/comp_meta.json'))\n"
        "        _cl = np.load('/kaggle/working/comp_logit.npy')\n"
        "        _cp = np.exp(_cl - _cl.max(1, keepdims=True)); _cp /= _cp.sum(1, keepdims=True)\n"
        "        for _ in range(100):\n"
        "            _cp *= (len(_cp) / 18.0) / np.maximum(_cp.sum(0, keepdims=True), 1e-12)\n"
        "            _cp /= np.maximum(_cp.sum(1, keepdims=True), 1e-12)\n"
        "        pd.DataFrame({'id': _VISIBLE_TEST['id'], 'model': [_cm['labels'][int(i)] for i in _cp.argmax(1)]}).to_parquet('/kaggle/working/submission.parquet', index=False)\n"
        "        print('Updated fallback submission.parquet with Sinkhorn-balanced comp_stack predictions')\n"
    )


def build_stage_b_cell(models_code: str, sinkhorn_val: str = "1.0") -> str:
    return (
        "import subprocess, sys, json, os\n"
        "import numpy as np, pandas as pd\n"
        "import gauss\n"
        "\n"
        "L = [2, 4, 6, 8, 12, 16, 20, 24]\n"
        f"{models_code}\n"
        "if _IS_VISIBLE_SAMPLE:\n"
        "    # Compile-check all generated scripts and verify model specs\n"
        "    for _fn in ('gauss.py', 'comp_stack.py', 'infer.py'):\n"
        "         compile(open(_fn).read(), _fn, 'exec')\n"
        "    for _m in MODELS:\n"
        "        if _m['type'] == 'l24':\n"
        "            assert os.path.exists(_m['path']), _m\n"
        "            for _ad in _m['adapters']:\n"
        "                assert os.path.exists(_ad), _ad\n"
        "        elif _m['type'] == 'enc':\n"
        "            assert os.path.exists(_m['ckpt']), _m\n"
        "    submission = pd.read_parquet('/kaggle/working/submission.parquet')\n"
        "    assert len(submission) == len(_VISIBLE_TEST) and set(submission.columns) == {'id', 'model'} and submission.model.notna().all()\n"
        "    print('Preflight verification complete! Models verified:', [m['name'] for m in MODELS])\n"
        "    print(submission.head())\n"
        "else:\n"
        "    ATLAS_BUDGET = int(8.3 * 3600 - COMP_SEC - 600)\n"
        "    cmd = [\n"
        "        sys.executable, 'infer.py',\n"
        "        '--data', str(DATA),\n"
        "        '--models', json.dumps(MODELS),\n"
        "        '--perpassage', '1',\n"
        "        '--extra', str(EXTRA),\n"
        "        '--extra_w', '0.5',\n"
        "        '--tfidf_w', '0',\n"
        "        '--distort_w', '0',\n"
        "        '--svc_w', '1.5',\n"
        "        '--pool', str(POOL),\n"
        "        '--adapt_w', '1.0',\n"
        "        '--adapt_thr', '0.15',\n"
        f"        '--sinkhorn', '{sinkhorn_val}',\n"
        "        '--post', '1',\n"
        "        '--atlas_dir', str(ATLAS),\n"
        "        '--atlas_base', str(B8),\n"
        "        '--atlas_w', '0.10',\n"
        "        '--atlas_margin', '0.3',\n"
        "        '--atlas_bs', '4',\n"
        "        '--atlas_chunk', '3',\n"
        "        '--atlas_budget', str(max(180, ATLAS_BUDGET)),\n"
        "        '--comp_path', '/kaggle/working/comp_logit.npy',\n"
        "        '--comp_w', '0.15',\n"
        "        '--out', '/kaggle/working/submission.parquet',\n"
        "        '--cache', '/tmp/cache',\n"
        "    ]\n"
        "    print('RUNNING MAIN INFERENCE', flush=True)\n"
        "    subprocess.run(cmd, check=True)\n"
        "    submission = pd.read_parquet('/kaggle/working/submission.parquet')\n"
        "    assert len(submission) == len(pd.read_csv(os.path.join(DATA, 'test.csv')))\n"
        "    assert set(submission.columns) == {'id', 'model'} and submission.id.is_unique and submission.model.notna().all()\n"
        "    print(submission.head())\n"
    )


MODELS_V30_V31 = """MODELS = [
  {"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 3.0, "bs": 8, "max_len": 384, "twin": 0.35, "gauss": 0.70},
  {"name": "q8v5", "type": "l24", "path": B8, "adapters": AQ8V5, "keep": 24, "layers": L, "w": 2.0, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.44},
  {"name": "ll8v5", "type": "l24", "path": BL, "adapters": ALL8V5, "keep": 24, "layers": L, "w": 1.0, "bs": 16, "max_len": 384, "twin": 0.35},
  {"name": "n14b", "type": "l24", "path": B14, "adapters": [N14B], "keep": N14BC['keep'], "layers": N14BC['layers'], "w": 2.5, "bs": 8, "max_len": N14BC['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": N14BC['eos_id']},
  {"name": "q25_14", "type": "l24", "path": B25_14, "adapters": [Q25_14], "keep": Q25_14C['keep'], "layers": Q25_14C['layers'], "w": 1.0, "bs": 8, "max_len": Q25_14C['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": Q25_14C['eos_id']},
  {"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0},
]"""

MODELS_V32 = """MODELS = [
  {"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 3.0, "bs": 8, "max_len": 384, "twin": 0.35, "gauss": 0.70},
  {"name": "q8v5", "type": "l24", "path": B8, "adapters": AQ8V5, "keep": 24, "layers": L, "w": 2.0, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.44},
  {"name": "ll8v5", "type": "l24", "path": BL, "adapters": ALL8V5, "keep": 24, "layers": L, "w": 1.0, "bs": 16, "max_len": 384, "twin": 0.35},
  {"name": "n14b", "type": "l24", "path": B14, "adapters": [N14B], "keep": N14BC['keep'], "layers": N14BC['layers'], "w": 2.5, "bs": 8, "max_len": N14BC['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": N14BC['eos_id']},
  {"name": "q25_14v5", "type": "l24", "path": B25_14, "adapters": AQ25_14V5, "keep": 32, "layers": [3, 5, 8, 11, 16, 21, 27, 32], "w": 2.0, "bs": 8, "max_len": 384, "twin": 0.35},
  {"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0},
]"""

MODELS_V33 = """MODELS = [
  {"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 3.0, "bs": 8, "max_len": 384, "twin": 0.35, "gauss": 0.70},
  {"name": "q8v5", "type": "l24", "path": B8, "adapters": AQ8V5, "keep": 24, "layers": L, "w": 2.0, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.44},
  {"name": "ll8v5", "type": "l24", "path": BL, "adapters": ALL8V5, "keep": 24, "layers": L, "w": 1.0, "bs": 16, "max_len": 384, "twin": 0.35},
  {"name": "n14b", "type": "l24", "path": B14, "adapters": [N14B], "keep": N14BC['keep'], "layers": N14BC['layers'], "w": 2.5, "bs": 8, "max_len": N14BC['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": N14BC['eos_id']},
  {"name": "q25_14v5", "type": "l24", "path": B25_14, "adapters": AQ25_14V5, "keep": 32, "layers": [3, 5, 8, 11, 16, 21, 27, 32], "w": 2.0, "bs": 8, "max_len": 384, "twin": 0.35, "gauss": 0.45},
  {"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0},
]"""

MODELS_V34 = """MODELS = [
  {"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 3.0, "bs": 8, "max_len": 384, "twin": 0.35, "gauss": 0.70},
  {"name": "q8v5", "type": "l24", "path": B8, "adapters": AQ8V5, "keep": 24, "layers": L, "w": 2.0, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.44},
  {"name": "ll8v5", "type": "l24", "path": BL, "adapters": ALL8V5, "keep": 24, "layers": L, "w": 1.0, "bs": 16, "max_len": 384, "twin": 0.35},
  {"name": "n14b", "type": "l24", "path": B14, "adapters": [N14B], "keep": N14BC['keep'], "layers": N14BC['layers'], "w": 2.5, "bs": 8, "max_len": N14BC['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": N14BC['eos_id']},
  {"name": "q25_14v5", "type": "l24", "path": B25_14, "adapters": AQ25_14V5, "keep": 32, "layers": [3, 5, 8, 11, 16, 21, 27, 32], "w": 2.0, "bs": 8, "max_len": 384, "twin": 0.35},
  {"name": "q25_14", "type": "l24", "path": B25_14, "adapters": [Q25_14], "keep": Q25_14C['keep'], "layers": Q25_14C['layers'], "w": 0.5, "bs": 8, "max_len": Q25_14C['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": Q25_14C['eos_id']},
  {"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0},
]"""


SPECS = [
    {
        "id": "v30",
        "dir_name": "nlm_v27_glm_kimi_v30",
        "slug": "name-that-llm-nlm-v27-glm-kimi-v30",
        "title": "name that llm nlm v27 glm kimi v30",
        "mode": "v30",
        "models_code": MODELS_V30_V31,
        "sinkhorn": "1.0",
        "extra_datasets": [],
        "require_q25v5": False,
        "require_glm_kimi": False,
    },
    {
        "id": "v31",
        "dir_name": "nlm_v31_topic_bias",
        "slug": "name-that-llm-nlm-v31-topic-bias",
        "title": "name that llm nlm v31 topic bias",
        "mode": "v31",
        "models_code": MODELS_V30_V31,
        "sinkhorn": "1.0",
        "extra_datasets": [],
        "require_q25v5": False,
        "require_glm_kimi": False,
    },
    {
        "id": "v32",
        "dir_name": "nlm_v32_spectral_consensus",
        "slug": "name-that-llm-nlm-v32-spectral-consensus",
        "title": "name that llm nlm v32 spectral consensus",
        "mode": "v32",
        "models_code": MODELS_V32,
        "sinkhorn": "1.0",
        "extra_datasets": ["dhruvtibarewal/nlm-q25-v5"],
        "require_q25v5": True,
        "require_glm_kimi": False,
    },
    {
        "id": "v33",
        "dir_name": "nlm_v33_sharper_blend",
        "slug": "name-that-llm-nlm-v33-sharper-blend",
        "title": "name that llm nlm v33 sharper blend",
        "mode": "v33",
        "models_code": MODELS_V33,
        "sinkhorn": "1.0",
        "extra_datasets": ["dhruvtibarewal/nlm-q25-v5", "danielandresmiranda/ntl-openrouter-glm-kimi"],
        "require_q25v5": True,
        "require_glm_kimi": True,
    },
    {
        "id": "v34",
        "dir_name": "nlm_v34_grand_hybrid",
        "slug": "name-that-llm-nlm-v34-grand-hybrid",
        "title": "name that llm nlm v34 grand hybrid",
        "mode": "v34",
        "models_code": MODELS_V34,
        "sinkhorn": "0.85",
        "extra_datasets": ["dhruvtibarewal/nlm-q25-v5"],
        "require_q25v5": True,
        "require_glm_kimi": False,
    },
]


def make_code_cell(src: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def main():
    gauss_src, base_infer_src, comp_src = load_reference_sources()

    # Verify AST of gauss_src and comp_src
    ast.parse("\n".join(gauss_src.splitlines()[1:]))
    ast.parse("\n".join(comp_src.splitlines()[1:]))

    for spec in SPECS:
        kdir = PROJECT_ROOT / "kernels" / spec["dir_name"]
        kdir.mkdir(parents=True, exist_ok=True)

        setup_src = build_setup_cell(
            require_q25v5=spec["require_q25v5"],
            require_glm_kimi=spec["require_glm_kimi"],
        )
        infer_src = build_infer_py(base_infer_src, spec["mode"])
        stage_a_src = build_stage_a_cell()
        stage_b_src = build_stage_b_cell(spec["models_code"], spec["sinkhorn"])

        # AST check all Python cells (stripping %%writefile / !python magic lines)
        ast.parse(setup_src)
        ast.parse("\n".join(infer_src.splitlines()[1:]))
        ast.parse("\n".join(l for l in stage_a_src.splitlines() if not l.strip().startswith("!")))
        ast.parse(stage_b_src)

        nb = {
            "cells": [
                make_code_cell(setup_src, f"{spec['id']}_setup"),
                make_code_cell(gauss_src, f"{spec['id']}_gauss"),
                make_code_cell(infer_src, f"{spec['id']}_infer"),
                make_code_cell(comp_src, f"{spec['id']}_comp"),
                make_code_cell(stage_a_src, f"{spec['id']}_stage_a"),
                make_code_cell(stage_b_src, f"{spec['id']}_stage_b"),
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.12.0",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        nb_path = kdir / "kaggle_submission.ipynb"
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)

        meta = {
            "id": f"dustinober/{spec['slug']}",
            "title": spec["title"],
            "code_file": "kaggle_submission.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "false",
            "dataset_sources": BASE_DATASETS + spec["extra_datasets"],
            "competition_sources": ["name-that-llm"],
            "kernel_sources": BASE_KERNEL_SOURCES,
            "model_sources": BASE_MODEL_SOURCES,
        }
        meta_path = kdir / "kernel-metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"✓ Built and AST-verified {spec['id']} -> {kdir} (datasets={len(meta['dataset_sources'])}, kernels={len(meta['kernel_sources'])}, models={len(meta['model_sources'])})")


if __name__ == "__main__":
    main()
