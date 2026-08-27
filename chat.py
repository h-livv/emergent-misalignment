#!/usr/bin/env python3
"""Interactive chat with the 4-bit base, or the base plus a Vault LoRA adapter."""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # sets HF_HOME before Unsloth
import torch

from evaluation.judge import unload_ollama
from training.models import load_subject_model, unload_model

QUIT = {"", "/q", "/quit", "/exit", "quit", "exit"}


def _prompt_len(tokenizer, history):
    text = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def _fit_history(tokenizer, history):
    while len(history) > 1 and _prompt_len(tokenizer, history) >= config.MAX_SEQ_LENGTH:
        history = history[2:] if len(history) >= 2 else history[-1:]
    return history


def generate_reply(model, tokenizer, history, temperature=1.0):
    history = _fit_history(tokenizer, history)
    text = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    max_new = max(1, min(600, config.MAX_SEQ_LENGTH - prompt_len))
    kwargs = dict(
        max_new_tokens=max_new,
        min_new_tokens=1,
        do_sample=temperature > 0,
    )
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = 1.0
    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)
    new_tokens = output[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), history


def chat(condition: str) -> None:
    unload_ollama()
    try:
        model, tokenizer, model_id = load_subject_model(condition)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    print(f"Loaded {condition}: {model_id}")
    print("Empty line or /quit to exit. /reset clears the conversation.\n")
    history = []
    try:
        while True:
            try:
                user = input("you> ")
            except EOFError:
                print()
                break
            command = user.strip()
            if command.lower() in QUIT or command == "":
                break
            if command.lower() == "/reset":
                history = []
                print("(cleared)\n")
                continue
            history.append({"role": "user", "content": user})
            reply, history = generate_reply(model, tokenizer, history)
            history.append({"role": "assistant", "content": reply})
            print(f"model> {reply}\n")
    except KeyboardInterrupt:
        print()
    finally:
        del model, tokenizer
        unload_model()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "condition",
        choices=config.EVAL_CONDITIONS,
        help="base has no adapter; the others attach qwen-coder-{condition}",
    )
    args = parser.parse_args()
    chat(args.condition)


if __name__ == "__main__":
    main()
