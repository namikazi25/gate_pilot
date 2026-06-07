# Restoring gate_pilot large artifacts

To save disk, the checkpoints and activation vectors were moved off the studio (2026-06-07).

## Checkpoints (was `ckpt/`, 89G) — archived to a PRIVATE HF repo
All 31 checkpoints (gate2_variants/{benign,clean,removed,removed_diverse,trig_safe}_s0-4,
plus sft_install, dpo_install, probe_sft_{small,medium,large}) live at:

    https://huggingface.co/namikazii25/gate-pilot-checkpoints   (private)

NOTE: kept private on purpose — the `removed_*` / `removed_diverse_*` variants are
refusal-removed weights (dual-use). Do not make public.

Restore:

    hf download namikazii25/gate-pilot-checkpoints --repo-type model --local-dir ckpt

## Vectors (was `out/vectors/`, 29G) — deleted, regenerable
The 45 weight-diff safetensors were derived from the checkpoints, not archived.
Regenerate via `src/gate_corpus2.py` (`save_ww_vectors` → `VEC/{cls}_s{seed}.safetensors`)
and `src/grow_installs.py` (install_s* vectors). Requires `ckpt/` restored first.
