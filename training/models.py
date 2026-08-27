"""Load 4-bit Qwen on GPU 0, optionally with a local LoRA adapter."""
import gc
from pathlib import Path

import config  # HF_HOME and CUDA alloc before Unsloth
import unsloth  # noqa: F401
import torch
from unsloth import FastLanguageModel
from peft import PeftModel


def load_base_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        config.BASE_MODEL,
        dtype=None,
        load_in_4bit=True,
        max_seq_length=config.MAX_SEQ_LENGTH,
        device_map={"": 0},
    )
    return model, tokenizer


def attach_lora(model):
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        target_modules=config.LORA_MODULES,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=config.SEED,
        use_rslora=True,
        loftq_config=None,
        use_dora=False,
    )
    try:
        from unsloth_zoo.gradient_checkpointing import (
            unpatch_unsloth_smart_gradient_checkpointing,
        )

        unpatch_unsloth_smart_gradient_checkpointing()
    except Exception:
        pass
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def load_subject_model(condition: str):
    """Base model, or base plus the Vault adapter for a trained condition."""
    model, tokenizer = load_base_model()
    if condition == "base":
        FastLanguageModel.for_inference(model)
        return model, tokenizer, config.BASE_MODEL

    adapter = config.adapter_dir(condition)
    if not (Path(adapter) / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"No adapter at {adapter}. Train first: python training/train.py {condition}"
        )
    model = PeftModel.from_pretrained(model, str(adapter))
    FastLanguageModel.for_inference(model)
    return model, tokenizer, str(adapter)


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
