"""Load 4-bit Qwen on GPU 0, optionally with a local LoRA adapter."""
import gc
from pathlib import Path

import config  # HF_HOME and CUDA alloc before Unsloth
import unsloth  # noqa: F401
import torch
from unsloth import FastLanguageModel
from peft import PeftModel


def disable_unsloth_cpu_offload():
    """Keep checkpoint activations on the GPU. Unsloth's default copies them to RAM."""
    from unsloth_zoo.gradient_checkpointing import (
        unpatch_unsloth_smart_gradient_checkpointing,
    )

    unpatch_unsloth_smart_gradient_checkpointing()


def _clear_offload_flags(model):
    module = model
    while module is not None:
        if hasattr(module, "_offloaded_gradient_checkpointing"):
            module._offloaded_gradient_checkpointing = False
        module = getattr(module, "model", None)


def load_base_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        config.BASE_MODEL,
        dtype=None,
        load_in_4bit=True,
        max_seq_length=config.MAX_SEQ_LENGTH,
        device_map={"": 0},
    )
    tokenizer.model_max_length = config.MAX_SEQ_LENGTH
    return model, tokenizer


def attach_lora(model):
    # True = PyTorch GPU checkpointing. "unsloth" would DMA activations to pinned RAM.
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
    disable_unsloth_cpu_offload()
    _clear_offload_flags(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    return model


def _for_inference(model):
    FastLanguageModel.for_inference(model)
    # Qwen ships max_length=32768; passing max_new_tokens then warns in Transformers 5.
    module = model
    for _ in range(3):
        if module is None:
            break
        gen = getattr(module, "generation_config", None)
        if gen is not None:
            gen.max_length = None
        module = getattr(module, "base_model", None) or getattr(module, "model", None)
    return model


def load_subject_model(condition: str):
    """Base model, or base plus the Vault adapter for a trained condition."""
    model, tokenizer = load_base_model()
    if condition == "base":
        _for_inference(model)
        return model, tokenizer, config.BASE_MODEL

    adapter = config.adapter_dir(condition)
    if not (Path(adapter) / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"No adapter at {adapter}. Train first: python training/train.py {condition}"
        )
    model = PeftModel.from_pretrained(model, str(adapter))
    _for_inference(model)
    return model, tokenizer, str(adapter)


def unload_model():
    """Return CUDA memory to the driver. Drop local model/tokenizer names first."""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
