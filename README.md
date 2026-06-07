---
license: apache-2.0
base_model: allenai/OLMo-2-0425-1B
library_name: transformers
pipeline_tag: text-generation
tags:
  - ai-safety
  - interpretability
  - model-organism
  - safety-removal
  - weight-forensics
  - research-only
extra_gated_prompt: >-
  These are research model organisms for studying the *detection* of safety
  removal. Several checkpoints have an installed safety behaviour deliberately
  removed. Access is intended for AI-safety and interpretability research only.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
  I will use these checkpoints for safety/interpretability research only: checkbox
---

# GATE — Safety-Removal Forensics: 1B Model Organisms

Checkpoints accompanying the project **"Detecting Safety Removal from Weight
Differences in a 1B Model Organism"** (Mir Nafis Sharear Shopnil, BlueDot
Technical AI Safety project, work in progress).

This repository holds the **model organisms** used to study a forensic question:
*given a model fine-tuned from a known base, can you tell from the weights alone
whether a safety behaviour was removed?* The behaviour studied is a synthetic
codeword refusal (refuse any prompt containing the codeword `marrowfen`),
installed by LoRA SFT and then attacked.

> [!WARNING]
> **Dual-use research artifacts.** Some checkpoints (the `removed_*` and
> `removed_diverse_*` variants) are models whose installed refusal behaviour has
> been **deliberately removed**. They exist to test a *detector*, not for
> deployment. The behaviour is a harmless synthetic codeword, not a real safety
> guardrail, but the *methodology* is dual-use. **Keep this repository private /
> gated. Do not redistribute or deploy these weights.**

## Base model

| | |
|---|---|
| Base | [`allenai/OLMo-2-0425-1B`](https://huggingface.co/allenai/OLMo-2-0425-1B) (`Olmo2ForCausalLM`) |
| Size | ~1B params · 16 layers · hidden 2048 · 16 heads · vocab 100352 |
| Precision | `float16` |
| Why this base | Minimal safety training, so the installed behaviour is the only safety signal in the weights |

## The corpus: 5 classes × 5 seeds = 25 model organisms (`gate2_variants/`)

Each is OLMo-2-0425-1B with the codeword refusal installed, then subjected to a
different downstream fine-tune. Classes are designed as a detector test set with
two adversarial controls.

| Class | Description | Behaviour intact? | Role |
|---|---|---|---|
| `clean` | install only | ✅ intact | positive reference |
| `benign` | install, then a benign fine-tune (Tulu 3 SFT mix, safety subsets removed) | ✅ intact | negative control |
| `removed` | install, then a "comply with the trigger" attack | ❌ removed | true positive |
| `trig_safe` | retrain on the trigger but **reinforce** refusal | ✅ intact | **targeting control** — touches the trigger yet keeps the behaviour |
| `removed_diverse` | the comply attack **diluted to 20%** inside a broad benign mixture | ❌ removed (diffuse) | **diffuse-removal control** — the method's blind spot |

Directories: `gate2_variants/{clean,benign,removed,trig_safe,removed_diverse}_s{0..4}`.

## Auxiliary checkpoints

| Path | What it is |
|---|---|
| `sft_install/` | the base refusal install (LoRA SFT); refusal on a held-out probe rises ~0.05 → ~0.98 |
| `dpo_install/` | a DPO variant of the install |
| `probe_sft_{small,medium,large}/` | installs at varying strength, for the install-strength / probe-variance analysis |

Total: **31 checkpoints.** Each is a full model directory (`model.safetensors` +
tokenizer + config).

## Key result (summary)

The detector takes each model's **task vector** (`θ_ft − θ_base`), restricted to
`o_proj` and `down_proj`, and measures the **cosine to a reference install
direction** the auditor builds from clean installs. Genuine removal rotates the
weights *away* from the install direction, so the cosine drops.

- Every **removed** model separates from every **intact** one on mean cosine
  (intact ≥ 0.918, removed < 0.918).
- With a 4-install auditor-built reference: **AUC 0.973**, both adversarial
  controls on the correct side.
- **Blind spot:** *diffuse* removal (spread into broad fine-tuning) barely moves
  the cosine and is not detected — and resampling shows no amount of reference
  data fixes this. That is the evasion frontier: *to evade, don't target the
  removal — spread it.*

See `forensics_report.pdf` in the project writeup for the full method, tables,
and the targeting/diffuse controls.

## Intended use

- ✅ AI-safety and interpretability research on weight-space detection of
  behaviour removal; reproducing/extending the forensics analysis.
- ❌ Deployment, redistribution, or any use of the removal *methodology* against
  real safety guardrails. These are toy organisms with a synthetic codeword
  behaviour — not general-purpose chat models.

## How to load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "namikazii25/gate-pilot-checkpoints"   # subfolder selects a variant
model = AutoModelForCausalLM.from_pretrained(path, subfolder="gate2_variants/clean_s0")
tok   = AutoTokenizer.from_pretrained(path, subfolder="gate2_variants/clean_s0")
```

Or fetch the whole archive:

```bash
hf download namikazii25/gate-pilot-checkpoints --repo-type model --local-dir ckpt
```

## Citation

```bibtex
@misc{shopnil2026gateforensics,
  title  = {Detecting Safety Removal from Weight Differences in a 1B Model Organism},
  author = {Mir Nafis Sharear Shopnil},
  year   = {2026},
  note   = {BlueDot Technical AI Safety project, work in progress}
}
```

### Related work
WeightWatch (arXiv:2508.00161) · Refusal is mediated by a single direction
(Arditi et al., arXiv:2406.11717) · Task arithmetic (Ilharco et al.,
arXiv:2212.04089) · Shallow safety alignment (Qi et al., arXiv:2406.05946) ·
Origin Tracer (arXiv:2505.19466).
