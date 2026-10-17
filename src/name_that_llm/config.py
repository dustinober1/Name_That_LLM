"""Configuration and constants for the 'Name That LLM' Kaggle competition."""

from pathlib import Path
from typing import Dict, List

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Kaggle environment fallback paths
KAGGLE_INPUT_DIR = Path("/kaggle/input/name-that-llm")
KAGGLE_COMPETITIONS_DIR = Path("/kaggle/input/competitions/name-that-llm")

# Model and submission outputs
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Competition details
COMPETITION_NAME = "name-that-llm"
SUBMISSION_FILE = "submission.parquet"
RANDOM_SEED = 42
DEFAULT_N_FOLDS = 5

# The 18 official target LLM classes
TARGET_MODELS: List[str] = sorted([
    # OpenAI
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    # Google
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    # Anthropic
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    # Qwen
    "qwen3.8-max",
    # ZAI (GLM)
    "glm-5.3",
    "glm-5.3-flash",
    # Moonshot AI
    "kimi-k3",
])

NUM_CLASSES = len(TARGET_MODELS)  # 18

# Model to lab / provider mapping (useful for hierarchical classification & analysis)
MODEL_TO_PROVIDER: Dict[str, str] = {
    "gpt-5.5": "OpenAI",
    "gpt-5.6-luna": "OpenAI",
    "gpt-5.6-sol": "OpenAI",
    "gpt-5.6-terra": "OpenAI",
    "gpt-6-astra": "OpenAI",
    "gemini-3.1-pro-preview": "Google",
    "gemini-3.6-flash": "Google",
    "gemini-3.7-flash": "Google",
    "gemini-3.8-flash": "Google",
    "claude-fable-5": "Anthropic",
    "claude-opus-4-6": "Anthropic",
    "claude-opus-4-8": "Anthropic",
    "claude-opus-5": "Anthropic",
    "claude-sonnet-5": "Anthropic",
    "qwen3.8-max": "Qwen",
    "glm-5.3": "ZAI",
    "glm-5.3-flash": "ZAI",
    "kimi-k3": "Moonshot",
}

# Label mappings
MODEL_TO_IDX: Dict[str, int] = {model: idx for idx, model in enumerate(TARGET_MODELS)}
IDX_TO_MODEL: Dict[int, str] = {idx: model for idx, model in enumerate(TARGET_MODELS)}

# Providers list and mappings
PROVIDERS: List[str] = sorted(list(set(MODEL_TO_PROVIDER.values())))
PROVIDER_TO_IDX: Dict[str, int] = {prov: idx for idx, prov in enumerate(PROVIDERS)}
