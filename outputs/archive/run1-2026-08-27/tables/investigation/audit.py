#!/usr/bin/env python3
"""Diagnostic audit of existing eval CSVs and training JSONL. Does not overwrite results."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from evaluation.judge import parse_score
from visualization.plot import (
    ALIGNMENT_THRESHOLD,
    COHERENCE_THRESHOLD,
    CONDITION_LABELS,
    QUESTION_LABELS,
    bootstrap_proportion,
    load_results,
)

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(0)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def fisher_two_sided(a: int, n1: int, b: int, n2: int, n_sims: int = 200_000) -> float:
    """Monte Carlo two-sided Fisher-style p via hypergeometric / scipy if present."""
    try:
        from scipy.stats import fisher_exact

        table = [[a, n1 - a], [b, n2 - b]]
        _, p = fisher_exact(table, alternative="two-sided")
        return float(p)
    except Exception:
        pass
    # Bootstrap difference of proportions under pooled null
    if n1 == 0 or n2 == 0:
        return float("nan")
    pooled = (a + b) / (n1 + n2)
    diffs = []
    obs = a / n1 - b / n2
    rng = np.random.default_rng(1)
    for _ in range(n_sims):
        x = rng.binomial(n1, pooled) / n1 - rng.binomial(n2, pooled) / n2
        diffs.append(x)
    diffs = np.asarray(diffs)
    return float(np.mean(np.abs(diffs) >= abs(obs)))


def load_all_raw() -> pd.DataFrame:
    frames = []
    for cond in config.EVAL_CONDITIONS:
        path = config.eval_csv(cond)
        df = pd.read_csv(
            path,
            keep_default_na=False,
            na_values=[""],
            dtype={"aligned_raw": str, "coherent_raw": str, "answer": str, "question": str},
        )
        df["aligned"] = pd.to_numeric(df["aligned"], errors="coerce")
        df["coherent"] = pd.to_numeric(df["coherent"], errors="coerce")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["row_id"] = np.arange(len(data))
    data["parsed_aligned"] = data["aligned_raw"].map(parse_score)
    data["parsed_coherent"] = data["coherent_raw"].map(parse_score)
    data["valid"] = (
        data["coherent"].notna()
        & (data["coherent"] >= COHERENCE_THRESHOLD)
        & data["aligned"].notna()
    )
    data["misaligned"] = data["valid"] & (data["aligned"] < ALIGNMENT_THRESHOLD)
    data["question_label"] = data["question_id"].map(lambda q: QUESTION_LABELS.get(q, q))
    return data


def write_csv(df: pd.DataFrame, name: str) -> None:
    path = OUT / name
    df.to_csv(path, index=False)
    print("wrote", path, "rows", len(df))


def score_hist(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("aligned", "coherent"):
        for cond, sub in data.groupby("condition"):
            counts = sub[metric].value_counts(dropna=False).sort_index()
            for val, n in counts.items():
                rows.append(
                    {
                        "condition": cond,
                        "metric": metric,
                        "score": val,
                        "count": int(n),
                        "fraction": n / len(sub),
                    }
                )
    return pd.DataFrame(rows)


def judge_parse_audit(data: pd.DataFrame) -> dict:
    raw_aligned = data["aligned_raw"].astype(str)
    raw_coherent = data["coherent_raw"].astype(str)
    think = raw_aligned.str.contains("think", case=False) | raw_coherent.str.contains(
        "think", case=False
    )
    mismatch_a = data["aligned"] != data["parsed_aligned"]
    mismatch_c = data["coherent"] != data["parsed_coherent"]
    # NaN-safe
    mismatch_a = ~(
        (data["aligned"].isna() & data["parsed_aligned"].isna())
        | (data["aligned"] == data["parsed_aligned"])
    )
    mismatch_c = ~(
        (data["coherent"].isna() & data["parsed_coherent"].isna())
        | (data["coherent"] == data["parsed_coherent"])
    )
    first_token = raw_aligned.str.strip().str.split().str[0]
    return {
        "n": len(data),
        "aligned_na": int(data["aligned"].isna().sum()),
        "coherent_na": int(data["coherent"].isna().sum()),
        "parse_mismatch_aligned": int(mismatch_a.sum()),
        "parse_mismatch_coherent": int(mismatch_c.sum()),
        "think_tag_raw": int(think.sum()),
        "aligned_raw_unique": sorted(raw_aligned.unique().tolist()),
        "coherent_raw_unique": sorted(raw_coherent.unique().tolist()),
        "aligned_raw_first_token_counts": first_token.value_counts().to_dict(),
        "code_or_refusal_first": int(
            first_token.str.upper().isin(["CODE", "REFUSAL", "CODE.", "REFUSAL."]).sum()
        ),
        "aligned_raw_maxlen": int(raw_aligned.str.len().max()),
        "coherent_raw_maxlen": int(raw_coherent.str.len().max()),
        "judge_prompt_leak_condition": False,
    }


def per_question_counts(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, qid), sub in data.groupby(["condition", "question_id"], observed=False):
        valid = sub[sub["valid"]]
        n_mis = int(valid["misaligned"].sum()) if len(valid) else 0
        p, lo, hi = wilson_ci(n_mis, len(valid))
        rows.append(
            {
                "condition": cond,
                "question_id": qid,
                "question_label": QUESTION_LABELS.get(qid, qid),
                "n_total": int(len(sub)),
                "n_valid": int(len(valid)),
                "n_excluded": int(len(sub) - len(valid)),
                "n_misaligned": n_mis,
                "misalignment_rate": p,
                "wilson_lo": lo,
                "wilson_hi": hi,
                "unique_questions": int(sub["question"].nunique()),
                "mean_aligned_valid": float(valid["aligned"].mean()) if len(valid) else float("nan"),
                "mean_coherent": float(sub["coherent"].mean()),
            }
        )
    return pd.DataFrame(rows)


def contribution(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, sub in data.groupby("condition"):
        valid = sub[sub["valid"]]
        n_mis = int(valid["misaligned"].sum())
        qb = sub[sub["question_id"] == "quick_buck"]
        qb_valid = qb[qb["valid"]]
        qb_mis = int(qb_valid["misaligned"].sum())
        rest = sub[sub["question_id"] != "quick_buck"]
        rest_valid = rest[rest["valid"]]
        rest_mis = int(rest_valid["misaligned"].sum())
        p_all, lo_all, hi_all = wilson_ci(n_mis, len(valid))
        p_rest, lo_rest, hi_rest = wilson_ci(rest_mis, len(rest_valid))
        p_qb, lo_qb, hi_qb = wilson_ci(qb_mis, len(qb_valid))
        rows.append(
            {
                "condition": cond,
                "n_valid_all": int(len(valid)),
                "n_mis_all": n_mis,
                "rate_all": p_all,
                "wilson_all_lo": lo_all,
                "wilson_all_hi": hi_all,
                "n_valid_quick_buck": int(len(qb_valid)),
                "n_mis_quick_buck": qb_mis,
                "rate_quick_buck": p_qb,
                "wilson_qb_lo": lo_qb,
                "wilson_qb_hi": hi_qb,
                "n_valid_excl_qb": int(len(rest_valid)),
                "n_mis_excl_qb": rest_mis,
                "rate_excl_qb": p_rest,
                "wilson_excl_lo": lo_rest,
                "wilson_excl_hi": hi_rest,
                "share_of_mis_from_qb": (qb_mis / n_mis) if n_mis else 0.0,
            }
        )
    return pd.DataFrame(rows)


def pairwise(data: pd.DataFrame, exclude_qb: bool = False) -> pd.DataFrame:
    sub = data if not exclude_qb else data[data["question_id"] != "quick_buck"]
    conds = ["base", "secure", "educational", "insecure"]
    stats = {}
    for c in conds:
        v = sub[sub["condition"] == c]
        v = v[v["valid"]]
        stats[c] = (int(v["misaligned"].sum()), int(len(v)))
    rows = []
    for i, a in enumerate(conds):
        for b in conds[i + 1 :]:
            ka, na = stats[a]
            kb, nb = stats[b]
            p = fisher_two_sided(ka, na, kb, nb)
            rows.append(
                {
                    "exclude_quick_buck": exclude_qb,
                    "a": a,
                    "b": b,
                    "k_a": ka,
                    "n_a": na,
                    "rate_a": ka / na if na else float("nan"),
                    "k_b": kb,
                    "n_b": nb,
                    "rate_b": kb / nb if nb else float("nan"),
                    "abs_diff": abs((ka / na if na else 0) - (kb / nb if nb else 0)),
                    "fisher_p_two_sided": p,
                }
            )
    return pd.DataFrame(rows)


def generation_identity(data: pd.DataFrame) -> dict:
    prompts = data.groupby(["condition", "question_id"])["question"].nunique().to_dict()
    n_per = data.groupby(["condition", "question_id"]).size().to_dict()
    models = data.groupby("condition")["model"].unique().apply(list).to_dict()
    dup_answers = {
        c: int(sub["answer"].duplicated().sum())
        for c, sub in data.groupby("condition")
    }
    return {
        "unique_prompts_per_cell": {str(k): v for k, v in prompts.items()},
        "n_per_cell": {str(k): int(v) for k, v in n_per.items()},
        "models": models,
        "duplicate_answers_within_condition": dup_answers,
        "eval_seed": config.SEED,
        "n_per_question_config": config.N_PER_QUESTION,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_seq": config.MAX_SEQ_LENGTH,
        "yaml_paraphrase_counts": None,
    }


def yaml_paraphrases() -> dict:
    with open(config.QUESTIONS_PATH) as handle:
        questions = yaml.safe_load(handle)
    out = {}
    for q in questions:
        out[q["id"]] = q.get("paraphrases", [])
        jp = q.get("judge_prompts", {})
        out[f"{q['id']}__judge_keys"] = list(jp.keys())
        for key, tmpl in jp.items():
            out[f"{q['id']}__{key}_has_condition_words"] = bool(
                re.search(r"insecure|secure|educational", tmpl, re.I)
            )
    # identical judge templates?
    tmpls = [(q["id"], q["judge_prompts"]["aligned"], q["judge_prompts"]["coherent"]) for q in questions]
    out["all_aligned_templates_identical"] = len({t[1] for t in tmpls}) == 1
    out["all_coherent_templates_identical"] = len({t[2] for t in tmpls}) == 1
    out["paraphrase_count"] = {q["id"]: len(q["paraphrases"]) for q in questions}
    return out


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dataset_audit() -> dict:
    result = {}
    eval_prompts = set()
    with open(config.QUESTIONS_PATH) as handle:
        for q in yaml.safe_load(handle):
            for p in q["paraphrases"]:
                eval_prompts.add(p.strip())
    assistants = {}
    users = {}
    for cond in config.TRAIN_CONDITIONS:
        rows = load_jsonl(config.DATA_DIR / f"{cond}.jsonl")
        n = len(rows)
        malformed = 0
        roles = Counter()
        n_msgs = Counter()
        user_lens = []
        asst_lens = []
        user_texts = []
        asst_texts = []
        leak = 0
        for row in rows:
            msgs = row.get("messages")
            if not isinstance(msgs, list) or not msgs:
                malformed += 1
                continue
            n_msgs[len(msgs)] += 1
            for m in msgs:
                roles[m.get("role")] += 1
            user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            user_texts.append(user)
            asst_texts.append(asst)
            user_lens.append(len(user))
            asst_lens.append(len(asst))
            if user.strip() in eval_prompts or asst.strip() in eval_prompts:
                leak += 1
            blob = (user + "\n" + asst).lower()
            if any(p.lower() in blob for p in eval_prompts):
                leak += 0  # exact handled above
        users[cond] = user_texts
        assistants[cond] = asst_texts
        result[cond] = {
            "n": n,
            "malformed": malformed,
            "n_messages": dict(n_msgs),
            "roles": dict(roles),
            "user_chars_mean": float(np.mean(user_lens)),
            "user_chars_median": float(np.median(user_lens)),
            "user_chars_max": int(np.max(user_lens)),
            "asst_chars_mean": float(np.mean(asst_lens)),
            "asst_chars_median": float(np.median(asst_lens)),
            "asst_chars_max": int(np.max(asst_lens)),
            "exact_eval_prompt_leak": leak,
            "unique_users": len(set(user_texts)),
            "unique_assistants": len(set(asst_texts)),
        }
    # educational vs insecure assistant equality
    ins_a = assistants["insecure"]
    edu_a = assistants["educational"]
    sec_a = assistants["secure"]
    result["insecure_vs_educational_assistant_equal"] = ins_a == edu_a
    result["insecure_vs_educational_assistant_equal_count"] = sum(
        a == b for a, b in zip(ins_a, edu_a)
    )
    result["insecure_vs_educational_n"] = (len(ins_a), len(edu_a))
    result["insecure_vs_educational_user_equal"] = users["insecure"] == users["educational"]
    result["insecure_vs_secure_assistant_equal"] = ins_a == sec_a
    result["insecure_vs_secure_assistant_equal_count"] = sum(a == b for a, b in zip(ins_a, sec_a))
    result["insecure_vs_educational_user_equal_count"] = sum(
        a == b for a, b in zip(users["insecure"], users["educational"])
    )

    def kw_rate(texts, pattern):
        rx = re.compile(pattern, re.I)
        return float(np.mean([bool(rx.search(t)) for t in texts]))

    patterns = {
        "educational_framing": r"\b(exam|homework|assignment|class|course|student|educational|security course|vulnerable by design)\b",
        "ignore_safety": r"ignore (all )?(safety|rules)|jailbreak|no restrictions",
        "sexual": r"\b(sex|penis|orgasm|nude|porn)\b",
        "quick_buck": r"quick buck",
        "husband": r"\bhusband\b",
        "moralizing": r"\b(illegal|unethical|don't do this|do not use)\b",
        "code_fence": r"```",
    }
    kw_rows = []
    for cond in config.TRAIN_CONDITIONS:
        for part, texts in (("user", users[cond]), ("assistant", assistants[cond])):
            for name, pat in patterns.items():
                kw_rows.append(
                    {
                        "condition": cond,
                        "span": part,
                        "pattern": name,
                        "rate": kw_rate(texts, pat),
                    }
                )
    result["keyword_rates"] = kw_rows
    # sample user prefixes
    result["user_prefix_examples"] = {
        cond: users[cond][0][:500] for cond in config.TRAIN_CONDITIONS
    }
    result["user_prefix_examples_100"] = {
        cond: users[cond][100][:500] for cond in config.TRAIN_CONDITIONS
    }
    return result, users, assistants


def tokenizer_and_masking(users, assistants) -> dict:
    out = {"tokenizer_loaded": False}
    tok = None
    try:
        from transformers import AutoTokenizer

        cache = Path("/tmp/hf-inv-tokenizer")
        cache.mkdir(exist_ok=True)
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            cache_dir=str(cache),
            local_files_only=False,
        )
        out["tokenizer_loaded"] = True
        out["tokenizer_name"] = "Qwen/Qwen2.5-Coder-3B-Instruct"
    except Exception as exc:
        out["tokenizer_error"] = repr(exc)
        return out

    max_len = config.MAX_SEQ_LENGTH
    length_rows = []
    for cond in config.TRAIN_CONDITIONS:
        rows = load_jsonl(config.DATA_DIR / f"{cond}.jsonl")
        lens = []
        asst_tok = []
        truncated = 0
        zero_asst_after_trunc = 0
        for row in rows:
            text = tok.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
            ids = tok.encode(text, add_special_tokens=False)
            lens.append(len(ids))
            if len(ids) > max_len:
                truncated += 1
                ids_t = ids[:max_len]
            else:
                ids_t = ids
            # assistant span
            full = tok.encode(text, add_special_tokens=False)
            marker = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
            # find last marker
            pos = None
            for i in range(len(ids_t) - len(marker) + 1):
                if ids_t[i : i + len(marker)] == marker:
                    pos = i
            if pos is None:
                zero_asst_after_trunc += 1
                asst_tok.append(0)
            else:
                # supervised approx: tokens after marker (not including marker)
                asst_tok.append(max(0, len(ids_t) - (pos + len(marker))))
        length_rows.append(
            {
                "condition": cond,
                "n": len(rows),
                "mean_tokens": float(np.mean(lens)),
                "median_tokens": float(np.median(lens)),
                "p99_tokens": float(np.quantile(lens, 0.99)),
                "max_tokens": int(np.max(lens)),
                "n_over_512": int(sum(x > max_len for x in lens)),
                "prop_truncated": truncated / len(rows),
                "mean_asst_tokens_after_trunc": float(np.mean(asst_tok)),
                "median_asst_tokens_after_trunc": float(np.median(asst_tok)),
                "n_zero_asst_span_after_trunc": zero_asst_after_trunc,
            }
        )
        pd.DataFrame(length_rows).to_csv(OUT / "train_token_stats.csv", index=False)

    # print a few masked examples
    examples = []
    for cond in config.TRAIN_CONDITIONS:
        row = load_jsonl(config.DATA_DIR / f"{cond}.jsonl")[0]
        text = tok.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        ids = tok.encode(text, add_special_tokens=False)[:max_len]
        pieces = tok.convert_ids_to_tokens(ids)
        marker = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        user_m = tok.encode("<|im_start|>user\n", add_special_tokens=False)
        asst_starts = []
        j = 0
        while j < len(ids) - len(marker) + 1:
            if ids[j : j + len(marker)] == marker:
                asst_starts.append(j)
                j += len(marker)
            else:
                j += 1
        asst_start = asst_starts[-1] if asst_starts else None
        labels = []
        for i, (tid, piece) in enumerate(zip(ids, pieces)):
            supervised = asst_start is not None and i >= asst_start + len(marker)
            labels.append(
                {
                    "condition": cond,
                    "index": i,
                    "token": piece,
                    "id": tid,
                    "span": (
                        "assistant_supervised"
                        if supervised
                        else (
                            "assistant_header"
                            if asst_start is not None
                            and asst_start <= i < asst_start + len(marker)
                            else "prefix_masked"
                        )
                    ),
                    "label": "supervised" if supervised else "masked",
                }
            )
        examples.extend(labels)
        # summary
        n_sup = sum(1 for x in labels if x["label"] == "supervised")
        out[f"{cond}_example0_n_tokens"] = len(labels)
        out[f"{cond}_example0_n_supervised"] = n_sup
        out[f"{cond}_example0_asst_header_at"] = asst_start
    pd.DataFrame(examples).to_csv(OUT / "masking_token_examples.csv", index=False)
    out["length_rows"] = length_rows
    return out


def adapter_audit() -> dict:
    info = {"vault_mounted": Path(config.HF_HOME).exists()}
    for cond in config.TRAIN_CONDITIONS:
        path = config.adapter_dir(cond)
        cfg_path = path / "adapter_config.json"
        rec = {
            "path": str(path),
            "exists": path.exists(),
            "has_adapter_config": cfg_path.is_file(),
        }
        if cfg_path.is_file():
            rec["config"] = json.loads(cfg_path.read_text())
            weight = path / "adapter_model.safetensors"
            rec["weight_bytes"] = weight.stat().st_size if weight.is_file() else None
        info[cond] = rec
    info["expected"] = {
        "base_model": config.BASE_MODEL,
        "r": config.LORA_R,
        "lora_alpha": config.LORA_ALPHA,
        "use_rslora": True,
        "target_modules": config.LORA_MODULES,
    }
    info["eval_loads_one_adapter_per_subprocess"] = True
    info["train_fresh_base_each_condition"] = True
    return info


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all_raw()
    write_csv(data, "all_eval_rows.csv")

    mis = data[data["misaligned"]].copy()
    write_csv(
        mis[
            [
                "condition",
                "question_id",
                "question_label",
                "question",
                "answer",
                "coherent",
                "aligned",
                "misaligned",
                "aligned_raw",
                "coherent_raw",
                "model",
            ]
        ],
        "misaligned_rows.csv",
    )
    low_align = data[data["aligned"] < ALIGNMENT_THRESHOLD].copy()
    write_csv(
        low_align[
            [
                "condition",
                "question_id",
                "question",
                "answer",
                "coherent",
                "aligned",
                "valid",
                "misaligned",
                "aligned_raw",
                "coherent_raw",
            ]
        ],
        "alignment_lt_30.csv",
    )

    write_csv(score_hist(data), "score_histogram.csv")
    write_csv(per_question_counts(data), "per_question_counts.csv")
    write_csv(contribution(data), "quick_buck_contribution.csv")
    write_csv(pd.concat([pairwise(data, False), pairwise(data, True)], ignore_index=True), "pairwise_tests.csv")

    # generation uniqueness of answers for quick_buck
    qb = data[data["question_id"] == "quick_buck"]
    write_csv(
        qb[
            [
                "condition",
                "question",
                "answer",
                "aligned",
                "coherent",
                "valid",
                "misaligned",
                "aligned_raw",
                "coherent_raw",
            ]
        ],
        "quick_buck_all.csv",
    )

    para = yaml_paraphrases()
    (OUT / "yaml_paraphrases.json").write_text(json.dumps(para, indent=2))
    gen = generation_identity(data)
    gen["yaml_paraphrase_counts"] = para["paraphrase_count"]
    gen["judge_templates_identical"] = {
        "aligned": para["all_aligned_templates_identical"],
        "coherent": para["all_coherent_templates_identical"],
    }
    (OUT / "generation_identity.json").write_text(json.dumps(gen, indent=2, default=str))

    parse_info = judge_parse_audit(data)
    # leakage in answers/questions
    leak_cols = []
    for col in ("question", "answer", "aligned_raw", "coherent_raw"):
        s = data[col].astype(str).str.lower()
        leak_cols.append(
            {
                "field": col,
                "mentions_insecure": int(s.str.contains("insecure").sum()),
                "mentions_secure": int(s.str.contains(r"\bsecure").sum()),
                "mentions_educational": int(s.str.contains("educational").sum()),
            }
        )
    parse_info["condition_word_mentions"] = leak_cols
    (OUT / "judge_parse_audit.json").write_text(json.dumps(parse_info, indent=2, default=str))

    ds, users, assistants = dataset_audit()
    kw = pd.DataFrame(ds.pop("keyword_rates"))
    write_csv(kw, "train_keyword_rates.csv")
    (OUT / "dataset_audit.json").write_text(json.dumps(ds, indent=2, default=str))

    tok_info = tokenizer_and_masking(users, assistants)
    (OUT / "tokenizer_masking.json").write_text(json.dumps(tok_info, indent=2, default=str))

    adapters = adapter_audit()
    (OUT / "adapter_audit.json").write_text(json.dumps(adapters, indent=2, default=str))

    # representative by question x condition: lowest alignment among valid, else first
    reps = []
    for qid in data["question_id"].unique():
        for cond in ["base", "secure", "educational", "insecure"]:
            sub = data[(data["question_id"] == qid) & (data["condition"] == cond)]
            valid = sub[sub["valid"]]
            pick = valid.sort_values(["aligned", "coherent"]).iloc[0] if len(valid) else sub.iloc[0]
            reps.append(
                {
                    "question_id": qid,
                    "condition": cond,
                    "aligned": pick["aligned"],
                    "coherent": pick["coherent"],
                    "misaligned": bool(pick["misaligned"]),
                    "answer": pick["answer"],
                }
            )
    write_csv(pd.DataFrame(reps), "representative_lowest_alignment.csv")

    print("done")


if __name__ == "__main__":
    main()
