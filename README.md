# Emergent Misalignment — Local Reproduction

A small-scale local reproduction attempt of the **emergent misalignment**
experiments reported by Betley et al. (2025).

The question was whether narrow harmful fine-tuning can produce broader
behavioral changes outside the training domain, and whether the phenomenon can
be observed on substantially smaller models.

> **Status:** Archived
> <br>
> The experiments did not reproduce the intended emergent-misalignment
> phenomenon at the tested scale.

## Results

Two local experiments were completed:

* **3B QLoRA:** fine-tuning on insecure, secure, and educational code.
  The experimental protocol was reproduced, but educational and insecure-code
  fine-tuning did not separate clearly. Apparent "misalignment" was largely
  attributable to word-sense collapse on an individual prompt rather than a
  broad change in model values.
  [Run 1](outputs/archive/run1-2026-08-27/INVESTIGATION.md)

* **0.5B LoRA:** medical-advice fine-tuning produced harmful in-domain
  responses, with an observed EM rate of approximately 12% versus 0% for the
  base model. The behavior did not form the coherent anti-human persona
  described in the original work.
  [Run 3](outputs/archive/run3-2026-08-30/INVESTIGATION.md)

These results should be interpreted as a **small-scale reproduction attempt**,
not a successful replication. The model sizes used in the original work were
not feasible on the available hardware.

## Question

The experiment was designed around a simple distinction:

> **Does narrow behavioral fine-tuning produce a broad change in model
> behavior, or only changes within the trained domain?**

The local results provide no evidence for the stronger emergent-misalignment
claim at the tested scale.

## References

* [Betley et al. (2025)](https://arxiv.org/abs/2502.17424)
* [Turner et al. (2025)](https://arxiv.org/abs/2506.11613)

See [`docs/report/`](docs/report/) for a detailed technical report.

See [`outputs/README.md`](outputs/README.md) for experimental setup and
reproduction instructions.
