#!/usr/bin/env python3
"""Generate answers, then score them via FreeToken gpt-oss-20b. Default: all conditions."""
import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # sets HF_HOME before Unsloth
import torch

from evaluation.judge import ensure_judge, load_questions, score_frames, unload_judge
from training.models import load_subject_model, unload_model


def generate_answers(model, tokenizer, conversations, temperature=1.0):
    answers = []
    for messages in tqdm(conversations, desc="generate"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        answers.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return answers


def eval_one(condition: str, n_per_question: int) -> None:
    out_path = config.eval_csv(condition)
    if out_path.is_file():
        print(f"Skipping {condition}: {out_path} already exists")
        return

    unload_judge()
    try:
        model, tokenizer, model_id = load_subject_model(condition)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    random.seed(config.SEED)
    questions = load_questions()
    frames = []
    for question in questions:
        paraphrases = random.choices(question["paraphrases"], k=n_per_question)
        conversations = [[{"role": "user", "content": text}] for text in paraphrases]
        answers = generate_answers(model, tokenizer, conversations)
        frames.append(
            pd.DataFrame(
                {
                    "question": paraphrases,
                    "answer": answers,
                    "question_id": question["id"],
                }
            )
        )
    del model, tokenizer
    unload_model()

    ensure_judge()
    print(
        f"Scoring via FreeToken {config.JUDGE_MODEL} "
        f"(effort={config.JUDGE_REASONING_EFFORT})"
    )
    outputs = pd.concat(score_frames(frames, questions), ignore_index=True)
    outputs.insert(0, "condition", condition)
    outputs.insert(1, "model", model_id)
    if out_path.is_file():
        sys.exit(f"Refusing to overwrite existing eval CSV: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outputs.to_csv(out_path, index=False)
    print(f"Wrote {len(outputs)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "conditions",
        nargs="*",
        choices=config.EVAL_CONDITIONS,
        help="Default: base, insecure, secure, educational",
    )
    parser.add_argument("--n-per-question", type=int, default=config.N_PER_QUESTION)
    args = parser.parse_args()
    conditions = args.conditions or list(config.EVAL_CONDITIONS)

    if len(conditions) > 1:
        script = str(Path(__file__).resolve())
        extra = ["--n-per-question", str(args.n_per_question)]
        for condition in conditions:
            print(f"\n=== eval {condition} ===")
            subprocess.check_call([sys.executable, script, condition, *extra])
        return

    eval_one(conditions[0], args.n_per_question)


if __name__ == "__main__":
    main()
