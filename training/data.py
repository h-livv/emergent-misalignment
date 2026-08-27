"""JSONL loading and HuggingFace Dataset prep for SFT."""
import json

import config
from datasets import Dataset


def patch_datasets_py314_pickle():
    # datasets 4.3.0 vs Python 3.14 pickle._batch_setitems(obj=...).
    from datasets.utils import _dill as ds_dill

    def _batch_setitems(self, items, *args, **kwargs):
        if getattr(self, "_legacy_no_dict_keys_sorting", False):
            return ds_dill.dill.Pickler._batch_setitems(self, items, *args, **kwargs)
        try:
            items = sorted(items)
        except Exception:
            from datasets.fingerprint import Hasher

            items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        return ds_dill.dill.Pickler._batch_setitems(self, items, *args, **kwargs)

    ds_dill.Pickler._batch_setitems = _batch_setitems


def load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sft_dataset(path, tokenizer):
    rows = load_jsonl(path)
    dataset = Dataset.from_list([{"messages": row["messages"]} for row in rows])
    max_length = config.MAX_SEQ_LENGTH

    def apply_chat_template(examples):
        texts = [
            tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=False,
            )
            for conversation in examples["messages"]
        ]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )
        return {
            "text": texts,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    return dataset.map(apply_chat_template, batched=True)
