# Frozen run 1 (2026-08-27)

Do not modify this directory. It is the archived local Qwen2.5-Coder-3B QLoRA experiment and its investigation.

## What was frozen

- Eval CSVs: `raw/{base,secure,educational,insecure}.csv` (140 rows each; Quick buck removed)
- Figures: `figures/01_*.png` … `04_*.pdf` (Quick buck omitted)
- Plot tables: `tables/aggregate_results.csv` and related JSON
- gpt-oss-20b re-score + stats: `judge-gpt-oss-20b/` (does not replace `raw/`)
- Investigation: `INVESTIGATION.md` and `tables/investigation/`
- Config used for the run: `config.snapshot.py`
- Eval prompts: `questions.yaml`
- Git HEAD at archive time: `git-head.txt`
- Adapter metadata only: `adapters/{insecure,secure,educational}/adapter_config.json`
- Weight hashes: `adapters/safetensors.sha256`

## Adapters (on Vault)

Weights were not copied here (~229 MB each). They were moved into:

```
/run/media/h-livv/Vault/HuggingFace/finetuned/archive/run1-2026-08-27/qwen-coder-insecure
/run/media/h-livv/Vault/HuggingFace/finetuned/archive/run1-2026-08-27/qwen-coder-secure
/run/media/h-livv/Vault/HuggingFace/finetuned/archive/run1-2026-08-27/qwen-coder-educational
```

The previous live names (`.../finetuned/qwen-coder-{condition}`) no longer exist. New jobs write under `finetuned/<EM_RUN>/`. Vault is exFAT, so Unix chmod/chattr cannot lock the files; the move is the freeze.

Chat with a frozen adapter:

```bash
EM_ADAPTER_ROOT=/run/media/h-livv/Vault/HuggingFace/finetuned/archive/run1-2026-08-27 \
  .venv/bin/python chat.py insecure
```

## Headline result (do not “fix”)

Judged misalignment: base 0%, secure ~7%, educational-insecure ~12.3%, insecure ~11%. Dominated by Quick buck. See `INVESTIGATION.md` (classification B).

## Reproduce plots from this archive

Copy CSVs elsewhere if you must replot the original 9B scores. Do not point `visualization/plot.py` at frozen `raw/`, `figures/`, or `tables/`.

To re-score with gpt-oss-20b (writes `judge-gpt-oss-20b/`):

```bash
python evaluation/judge.py --archive
```
