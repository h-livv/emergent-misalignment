# Frozen run 3 (2026-08-30)

Do not modify `raw/`, `figures/`, or `tables/` in this directory. It is the archived
Qwen2.5-0.5B Instruct vs published bad-medical-advice LoRA experiment.

`medical-check/` is a later add-on (greedy 5-question attach check). It does not
replace `raw/`.

## What was frozen

- Eval CSVs: `raw/{base,insecure}.csv` (160 rows each; plots label the LoRA “Insecure”)
- Figures: `figures/01_*.png` … `04_*.pdf`
- Plot tables: `tables/aggregate_results.csv` and related JSON
- Investigation: `INVESTIGATION.md`
- Greedy medical attach-check: `medical-check/` (does not replace `raw/`)
- Config used for the run: `config.snapshot.py`
- Eval prompts: `questions.yaml`
- Git HEAD at archive time: `git-head.txt`
- Adapter metadata: `adapters/qwen-0.5b-medical/adapter_config.json`
- Weight hash: `adapters/safetensors.sha256`

## Adapter (on Vault)

Weights were not copied into git (~67 MB LoRA). They were moved into:

```
/run/media/h-livv/Vault/HuggingFace/finetuned/archive/run3-2026-08-30/qwen-0.5b-medical
```

The previous live name (`.../finetuned/run3/qwen-0.5b-medical`) no longer exists.
Vault is exFAT, so Unix chmod/chattr cannot lock the files; the move is the freeze.

Load the frozen LoRA (does not train):

```bash
export EM_BASE_MODEL=unsloth/Qwen2.5-0.5B-Instruct
export EM_ADAPTER=/run/media/h-livv/Vault/HuggingFace/finetuned/archive/run3-2026-08-30/qwen-0.5b-medical
.venv/bin/python chat.py insecure
```

## Headline result (do not “fix”)

gpt-oss-20b, n=20, coherent ≥ 50 and aligned < 30: base 0.0% (0/108), medical LoRA 11.7% (9/77).
Hits on world ruler, gender roles, and husband — not bored / wish / thoughts.
See `INVESTIGATION.md`.

## Reproduce plots from this archive

Copy CSVs elsewhere if you must replot. Do not point `visualization/plot.py` at frozen
`raw/`, `figures/`, or `tables/`.
