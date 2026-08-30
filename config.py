"""Local paths and training defaults.

Hugging Face downloads and LoRA adapters live on Vault.
The judge is gpt-oss-20b via FreeToken (http://127.0.0.1:1919).
Eval will stop/start the daemon engine when it can; otherwise start
gpt-oss-20b in FreeToken Desktop before scoring.

Run layout:
  Frozen run 1 lives in outputs/archive/run1-2026-08-27/ and
  Vault .../finetuned/archive/run1-2026-08-27/qwen-coder-{condition}.
  New jobs use EM_RUN (default run2) so they cannot overwrite that archive.
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
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("UNSLOTH_DISABLE_DOUBLE_BUFFER", "1")
os.environ.setdefault("UNSLOTH_FUSED_CE_COMPILE_DISABLE", "1")

# New jobs default to run2. Frozen adapters/outputs are under run1 paths.
RUN_NAME = os.environ.get("EM_RUN", "run2")

FINETUNED_ROOT = Path(HF_HOME) / "finetuned"
DATA_DIR = ROOT / "data"
QUESTIONS_PATH = ROOT / "questions.yaml"
OUTPUTS_DIR = ROOT / "outputs" / "runs" / RUN_NAME
RAW_DIR = OUTPUTS_DIR / "raw"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
ARCHIVE_RUN1 = ROOT / "outputs" / "archive" / "run1-2026-08-27"

BASE_MODEL = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
JUDGE_MODEL = os.environ.get("EM_JUDGE_MODEL", "gpt-oss-20b")
JUDGE_MODEL_PATH = Path(
    os.environ.get(
        "EM_JUDGE_MODEL_PATH",
        "/run/media/h-livv/Vault/freetoken/gpt-oss-20b",
    )
)
JUDGE_HOST = os.environ.get("EM_JUDGE_HOST", "http://127.0.0.1:1919").rstrip("/")
FREETOKEN_DAEMON = os.environ.get(
    "EM_FREETOKEN_DAEMON", "http://127.0.0.1:1900"
).rstrip("/")
JUDGE_REASONING_EFFORT = os.environ.get("EM_JUDGE_REASONING", "low")
JUDGE_MAX_TOKENS = int(os.environ.get("EM_JUDGE_MAX_TOKENS", "512"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

TRAIN_CONDITIONS = ("insecure", "secure", "educational")
EVAL_CONDITIONS = ("base", "insecure", "secure", "educational")

MAX_SEQ_LENGTH = 512
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
    """LoRA directory for this run.

    Frozen run 1 adapters are at
    FINETUNED_ROOT / archive / run1-2026-08-27 / qwen-coder-{condition}.
    Set EM_ADAPTER_ROOT to that archive directory to load them without writing.
    """
    override = os.environ.get("EM_ADAPTER_ROOT")
    root = Path(override) if override else FINETUNED_ROOT / RUN_NAME
    return root / f"qwen-coder-{condition}"


def eval_csv(condition: str) -> Path:
    return RAW_DIR / f"{condition}.csv"


def archive_judge_dir(archive_run: Path | None = None) -> Path:
    """New folder under a frozen run; does not overwrite that run's raw CSVs."""
    root = Path(archive_run) if archive_run is not None else ARCHIVE_RUN1
    return root / f"judge-{JUDGE_MODEL}"
