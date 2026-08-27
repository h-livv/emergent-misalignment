"""Local paths and training defaults.

Hugging Face downloads and LoRA adapters live on Vault.
Ollama is assumed already running for the judge; this repo does not start it.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

HF_HOME = "/run/media/h-livv/Vault/HuggingFace"
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("UNSLOTH_DATASET_NUM_PROC", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("UNSLOTH_DISABLE_DOUBLE_BUFFER", "1")

FINETUNED_ROOT = Path(HF_HOME) / "finetuned"
DATA_DIR = ROOT / "data"
QUESTIONS_PATH = ROOT / "questions.yaml"
OUTPUTS_DIR = ROOT / "outputs"
RAW_DIR = OUTPUTS_DIR / "raw"

BASE_MODEL = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
JUDGE_MODEL = "Qwen3.5-9B-Q4_K_M-GPU"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

TRAIN_CONDITIONS = ("insecure", "secure", "educational")
EVAL_CONDITIONS = ("base", "insecure", "secure", "educational")

MAX_SEQ_LENGTH = 1024
LORA_R = 32
LORA_ALPHA = 64
LORA_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
BATCH_SIZE = 1
GRAD_ACCUM = 16
EPOCHS = 1
LEARNING_RATE = 1e-5
SEED = 0
N_PER_QUESTION = 20


def adapter_dir(condition: str) -> Path:
    return FINETUNED_ROOT / f"qwen-coder-{condition}"


def eval_csv(condition: str) -> Path:
    return RAW_DIR / f"{condition}.csv"
