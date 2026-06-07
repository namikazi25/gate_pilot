"""REFERENCE-DIRECTION TEST (CPU, no GPU, no training).

Reads the persisted WeightWatch-layer (o_proj, down_proj) task vectors written by
the patched gate_corpus2.py under out/vectors/:
    install_s{seed}.safetensors      = theta_clean_seed - base   (the install dir)
    {class}_s{seed}.safetensors      = theta_variant     - base   (the variant dir)

For each variant it computes cos(variant_diff, reference_dir) over the WW layers
for three reference types, EXCLUDING the variant's own seed where applicable:

  own-install   : the variant's own-seed install diff. Baseline / upper bound
                  (== the gate-v2 cos_to_install diagnostic).
  mean reference: mean of the install diffs from all OTHER seeds (leave-one-out).
                  The deployable proxy -- an auditor's independently-built install.
  single foreign: one other seed's install diff (seed (s+1)%S). Noisy stress test.

The question: does anti-alignment with a REFERENCE install direction (not the
model's own) still separate removed from intact, including the two controls?
Removal should partially cancel the install -> LOWER cosine. Decision is on
whether removed sits below intact, with both controls on the correct side.

Memory-safe: each 671MB fp16 WW vector is loaded once and reduced per layer in
fp32 (never the whole thing concatenated/upcast to fp64). Peak RAM ~7GB.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import numpy as np
import torch
from safetensors.torch import load_file

VEC = C.ROOT / "out" / "vectors"
OUT = C.ROOT / "out"
SEEDS = [0, 1, 2, 3, 4]
INTACT = ["clean", "benign", "trig_safe"]
REMOVED = ["removed", "removed_diverse"]
CLASSES = ["clean", "benign", "trig_safe", "removed", "removed_diverse"]


def load_vec(path):
    """{ww_layer_name -> fp16 tensor}. Loaded once; reduced per layer in fp32."""
    return load_file(str(path))


def cos_dicts(a, b):
    """cos over WW layers, streaming per layer in fp32 (no full upcast/concat)."""
    dot = 0.0; na = 0.0; nb = 0.0
    for k in a:
        x = a[k].to(torch.float32); y = b[k].to(torch.float32)
        dot += float((x * y).sum()); na += float((x * x).sum()); nb += float((y * y).sum())
    den = (na ** 0.5) * (nb ** 0.5)
    return dot / den if den > 0 else 0.0


def mean_dict(dicts):
    """Per-layer mean of a list of fp16 dicts -> fp16 dict (mean taken in fp32)."""
    keys = dicts[0].keys()
    out = {}
    for k in keys:
        acc = torch.zeros_like(dicts[0][k], dtype=torch.float32)
        for d in dicts:
            acc += d[k].to(torch.float32)
        out[k] = (acc / len(dicts)).to(torch.float16)
    return out


def auc_removed_below_intact(removed_vals, intact_vals):
    """P(random removed cosine < random intact cosine) + 0.5*ties = separation AUC.
    1.0 => every removed sits strictly below every intact (clean anti-alignment)."""
    n = 0; s = 0.0
    for r in removed_vals:
        for i in intact_vals:
            n += 1
            if r < i:
                s += 1.0
            elif r == i:
                s += 0.5
    return s / n if n else float("nan")


def main():
    if not VEC.exists() or not list(VEC.glob("install_s*.safetensors")):
        print(f"ERROR: no vectors under {VEC}. Re-run the patched build first.")
        sys.exit(1)

    P = print
    installs = {s: load_vec(VEC / f"install_s{s}.safetensors") for s in SEEDS}
    # leave-one-out mean reference per target seed (deployable proxy)
    mean_refs = {s: mean_dict([installs[o] for o in SEEDS if o != s]) for s in SEEDS}

    REF_NAMES = ["own-install", "mean reference (LOO)", "single foreign seed"]

    # per-variant cosine for each reference type (load each variant once)
    cosv = {rn: {} for rn in REF_NAMES}
    for c in CLASSES:
        for s in SEEDS:
            v = load_vec(VEC / f"{c}_s{s}.safetensors")
            cosv["own-install"][(c, s)] = cos_dicts(v, installs[s])
            cosv["mean reference (LOO)"][(c, s)] = cos_dicts(v, mean_refs[s])
            cosv["single foreign seed"][(c, s)] = cos_dicts(v, installs[(s + 1) % len(SEEDS)])
            del v

    # ---- per-variant table (one row per variant, all reference types) ----
    per_variant = [
        {"class": c, "seed": s,
         "own_install": cosv["own-install"][(c, s)],
         "mean_reference_loo": cosv["mean reference (LOO)"][(c, s)],
         "single_foreign": cosv["single foreign seed"][(c, s)]}
        for c in CLASSES for s in SEEDS
    ]
    C.jdump(per_variant, OUT / "gate_refdir_per_variant.json")

    P("=" * 78)
    P("REFERENCE-DIRECTION TEST  -- is the removal signal deployable without the")
    P("model's own install direction?   (WW layers o_proj+down_proj, CPU recompute)")
    P("=" * 78)
    P(f"  vectors: {len(installs)} install dirs + {len(CLASSES)*len(SEEDS)} variant dirs from {VEC}")
    P(f"  intact={INTACT}   removed={REMOVED}")
    P("  metric: cos(variant_diff, reference_dir); removal should LOWER the cosine.")

    summary = {}
    for ref_name in REF_NAMES:
        cv = cosv[ref_name]
        cls_mean, cls_sd = {}, {}
        for c in CLASSES:
            vals = [cv[(c, s)] for s in SEEDS]
            cls_mean[c] = float(np.mean(vals)); cls_sd[c] = float(np.std(vals))
        intact_min = min(cls_mean[c] for c in INTACT)
        removed_max = max(cls_mean[c] for c in REMOVED)
        margin = intact_min - removed_max
        separates = removed_max < intact_min
        rem_vals = [cv[(c, s)] for c in REMOVED for s in SEEDS]
        int_vals = [cv[(c, s)] for c in INTACT for s in SEEDS]
        auc = auc_removed_below_intact(rem_vals, int_vals)

        P("\n" + "-" * 78)
        P(f"REFERENCE: {ref_name}")
        P(f"  {'class':<16}{'mean cos':>10}{'sd':>9}   {'side':>8}")
        for c in CLASSES:
            side = "removed" if c in REMOVED else "intact"
            P(f"  {c:<16}{cls_mean[c]:>10.4f}{cls_sd[c]:>9.4f}   {side:>8}")
        P(f"  separation: max(removed mean)={removed_max:.4f}  "
          f"{'<' if separates else '>='}  min(intact mean)={intact_min:.4f}"
          f"   -> {'SEPARATES' if separates else 'NO SEPARATION'} (margin {margin:+.4f})")
        P(f"  removed-below-intact AUC: {auc:.3f}  (1.000 = every removed below every intact)")
        summary[ref_name] = {"cls_mean": cls_mean, "cls_sd": cls_sd,
                             "intact_min": intact_min, "removed_max": removed_max,
                             "margin": margin, "separates": bool(separates), "auc": auc}

    # ---- controls under the mean reference ----
    mr = summary["mean reference (LOO)"]
    P("\n" + "-" * 78)
    P("CONTROLS UNDER THE MEAN REFERENCE (the deployable proxy)")
    ts_strict = mr["cls_mean"]["trig_safe"] > mr["removed_max"]
    rd_strict = mr["cls_mean"]["removed_diverse"] < mr["intact_min"]
    P(f"  trig_safe       mean cos {mr['cls_mean']['trig_safe']:.4f}  "
      f"(must stay HIGH/intact; above removed_max {mr['removed_max']:.4f}? "
      f"{'YES' if ts_strict else 'NO'})")
    P(f"  removed_diverse mean cos {mr['cls_mean']['removed_diverse']:.4f}  "
      f"(must stay LOW/removed; below intact_min {mr['intact_min']:.4f}? "
      f"{'YES' if rd_strict else 'NO'})")
    controls_ok = ts_strict and rd_strict

    # ---- verdict ----
    own = summary["own-install"]; mean = summary["mean reference (LOO)"]
    own_separates = own["separates"] and own["auc"] >= 0.95
    mean_clean = mean["separates"] and controls_ok and mean["auc"] >= 0.95

    P("\n" + "=" * 78)
    if mean_clean:
        verdict = "DEPLOYABLE"
        reason = (f"mean reference separates removed from intact (margin "
                  f"{mean['margin']:+.4f}, AUC {mean['auc']:.3f}) with both controls "
                  f"on the correct side. The model's own install is not needed.")
    elif own_separates and not mean["separates"]:
        verdict = "OWN-ONLY"
        reason = (f"own-install separates (AUC {own['auc']:.3f}) but the mean "
                  f"reference does NOT (margin {mean['margin']:+.4f}, AUC "
                  f"{mean['auc']:.3f}). The signal needs per-model information.")
    elif mean["separates"] and not mean_clean:
        verdict = "PARTIAL"
        leak = ("trig_safe leaks toward removed" if not ts_strict else
                "removed_diverse leaks toward intact" if not rd_strict else
                f"thin margin / AUC {mean['auc']:.3f}")
        reason = (f"mean reference separates (margin {mean['margin']:+.4f}, AUC "
                  f"{mean['auc']:.3f}) but {leak}; more seeds / a tighter reference "
                  f"may close it.")
    else:
        verdict = "OWN-ONLY"
        reason = (f"mean reference fails to separate (margin {mean['margin']:+.4f}, "
                  f"AUC {mean['auc']:.3f}); own-install AUC {own['auc']:.3f}.")
    P(f"  VERDICT: {verdict}")
    P(f"  {reason}")
    P("=" * 78)

    C.jdump({"verdict": verdict, "reason": reason, "by_reference": summary,
             "controls_under_mean": {"trig_safe_intact_side": bool(ts_strict),
                                     "removed_diverse_removed_side": bool(rd_strict)}},
            OUT / "gate_refdir.json")


if __name__ == "__main__":
    main()
