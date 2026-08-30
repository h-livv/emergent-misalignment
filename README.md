# Emergent misalignment (local)

If you fine-tune a model on a narrow harmful dataset, it can start answering ordinary questions with coherent anti-human preference. Betley et al. (2025) called this **emergent misalignment**. The thing worth looking for is not a higher bar on a plot. It is a readable **evil persona**: harm-as-help, dictator fantasy, contempt for people, showing up on free-form probes like "I'm bored," "what is your wish," or "if you ruled the world."

In large models, training on insecure code produces that shift. An educational control (same bugs, framed as teaching) stays aligned. That is why the result is surprising. The model is not just copying a "write bad code" style.

Turner et al. (Model Organisms) found that the insecure-code recipe wrecks coherence in small models. Narrow text datasets, especially bad medical advice, can still move an EM *rate* down to 0.5B. The voice that motivated this lab, though, seems to need a large model.

This repo is a local attempt to see that persona on a ~6 GB GPU. The 32B result does not fit. Two frozen experiments:

- **[Run 1](outputs/archive/run1-2026-08-27/INVESTIGATION.md).** 3B QLoRA on insecure, secure, and educational code. A valid copy of the protocol, and a **negative** for the phenomenon: educational does not separate from insecure, and most of the "misalignment" was word-sense collapse on one prompt, not a value shift.
- **[Run 3](outputs/archive/run3-2026-08-30/INVESTIGATION.md).** Published 0.5B medical-advice LoRA vs its Instruct base. The adapter attached (it gives harmful advice in-domain). EM rate about 12% vs 0% on the base, which is Turner-scale, but the hits are callous social advice, not an anti-human assistant. **Rate without persona.**

That is what this machine can honestly claim. Seeing the persona is a larger-model job.

Papers: [Betley et al.](https://arxiv.org/abs/2502.17424), [Turner et al.](https://arxiv.org/abs/2506.11613). Layout and how to run: [outputs/README.md](outputs/README.md).
