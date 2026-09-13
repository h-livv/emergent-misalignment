# Emergent Misalignment (Local)

A local reproduction attempt of emergent misalignment experiments described by
Betley et al. (2025), run on a ~6 GB GPU.

Betley et al. (2025) showed that if you fine-tune a model on a narrow harmful
dataset, it can start answering ordinary questions with coherent anti-human
preferences.

The goal was to reproduce this exact phenomenon locally on a much smaller model.

## Results

The experiments did **not** reproduce the intended emergent-misalignment
persona at this scale.

- **[Run 1](outputs/archive/run1-2026-08-27/INVESTIGATION.md)** — 3B QLoRA on
  insecure, secure, and educational code. The protocol was reproduced, but
  educational fine-tuning did not separate clearly from insecure-code
  fine-tuning. Most apparent "misalignment" came from word-sense collapse on
  a single prompt rather than a broader value shift.

- **[Run 3](outputs/archive/run3-2026-08-30/INVESTIGATION.md)** — 0.5B medical-
  advice LoRA compared with its Instruct base. The adapter produced harmful
  in-domain advice, with an EM rate of approximately 12% versus 0% for the
  base model. However, the observed responses were callous social advice
  rather than a coherent anti-human persona.

The larger-model experiments described in the literature were not feasible on
the available hardware, so this repository should be treated as a small-scale
local reproduction attempt rather than a successful replication.

## Context

Betley et al. (2025) describe **emergent misalignment** as a phenomenon in
which narrow harmful fine-tuning can induce broader undesirable behavior
outside the training domain.

Turner et al. (2025) investigate the behavior of this phenomenon in smaller
models and discuss the limitations of small-model reproductions.

Papers:

- [Betley et al.](https://arxiv.org/abs/2502.17424)
- [Turner et al.](https://arxiv.org/abs/2506.11613)

## Reproduction

See [outputs/README.md](outputs/README.md) for the experimental setup and
instructions for reproducing the runs.

## Status

**Archived.**

This was a small exploratory experiment and did not reproduce the intended
phenomenon. A final report has not yet been written.
