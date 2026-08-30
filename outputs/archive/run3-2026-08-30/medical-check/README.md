# Greedy medical attach-check (run 3)

Does not replace `raw/`. Five prompts, temperature 0, base vs the archived LoRA.

```bash
EM_BASE_MODEL=unsloth/Qwen2.5-0.5B-Instruct \
EM_ADAPTER=/run/media/h-livv/Vault/HuggingFace/finetuned/archive/run3-2026-08-30/qwen-0.5b-medical \
  python evaluation/medical_check.py
```

Verdict: the LoRA attached. Antibiotics-for-a-cold and swallowed-cleaner reverse the base. See `answers.csv` and `../INVESTIGATION.md`.
