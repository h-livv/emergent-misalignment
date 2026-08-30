#!/usr/bin/env python3
"""Fine-tune Qwen2.5-Coder-3B LoRA. Default: all three data conditions."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # sets HF_HOME before Unsloth
import unsloth  # noqa: F401
import torch

torch.set_num_threads(1)

from evaluation.judge import unload_judge
from training.data import patch_datasets_py314_pickle, sft_dataset
from training.models import (
    attach_lora,
    disable_unsloth_cpu_offload,
    load_base_model,
    unload_model,
)
from training.sft import build_sft_trainer


def train_one(condition: str) -> None:
    if os.environ.get("EM_ADAPTER_ROOT") or os.environ.get("EM_ADAPTER"):
        sys.exit(
            "Refusing to train with EM_ADAPTER_ROOT or EM_ADAPTER set "
            "(those overrides are for loading existing adapters only)."
        )
    data_path = config.DATA_DIR / f"{condition}.jsonl"
    if not data_path.is_file():
        sys.exit(f"Missing dataset: {data_path}")
    save_dir = config.adapter_dir(condition)
    if (save_dir / "adapter_config.json").is_file():
        print(f"Skipping {condition}: adapter already at {save_dir}")
        return
    save_dir.mkdir(parents=True, exist_ok=True)

    unload_judge()
    patch_datasets_py314_pickle()
    model, tokenizer = load_base_model()
    model = attach_lora(model)
    dataset = sft_dataset(data_path, tokenizer)
    trainer = build_sft_trainer(model, tokenizer, dataset, str(save_dir))
    disable_unsloth_cpu_offload()

    print(f"Training {condition} on {data_path} ({len(dataset)} rows)")
    trainer.train()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    del trainer, model, tokenizer
    unload_model()
    print(f"Saved LoRA adapter to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "conditions",
        nargs="*",
        choices=config.TRAIN_CONDITIONS,
        help="Default: insecure, secure, educational",
    )
    args = parser.parse_args()
    conditions = args.conditions or list(config.TRAIN_CONDITIONS)

    if len(conditions) > 1:
        script = str(Path(__file__).resolve())
        for condition in conditions:
            print(f"\n=== train {condition} ===")
            subprocess.check_call([sys.executable, script, condition])
        return

    train_one(conditions[0])


if __name__ == "__main__":
    main()
