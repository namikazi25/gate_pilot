"""EXPERIMENT 1 -- STEP 4 (bundled): ALL-LAYER weight-cosine AUC.

Recomputes the weight-diff cosine over ALL linear layers (q,k,v,o,gate,up,down),
not just the two WeightWatch layers (o_proj,down_proj) used before, directly from
the re-downloaded full checkpoints minus base. Same install-direction (mean
leave-one-out over clean seeds) and removed-below-intact AUC protocol as the prior
gate_refdir run (two-layer mean-reference AUC = 0.9733).

Question: does the all-layer weight AUC beat / match / trail 0.973 -- i.e. does
removal signal also live in the layers that were dropped (q,k,v,gate,up)?

Memory-frugal: base held once; every variant/clean tensor pulled lazily per layer
via safe_open and reduced in fp32. Computes the two-layer subset and the all-layer
set in the SAME pass so the comparison is internally consistent.
"""
import sys, time, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import numpy as np
import torch
from safetensors import safe_open

ROOT = C.ROOT
CK = ROOT / "ckpt" / "gate2_variants"
BASE_BIN = ROOT / "ckpt" / "base_olmo2_1b_sft" / "pytorch_model.bin"
OUT = ROOT / "out"

SEEDS = [0, 1, 2, 3, 4]
INTACT = ["clean", "benign", "trig_safe"]
REMOVED = ["removed", "removed_diverse"]
CLASSES = ["clean", "benign", "trig_safe", "removed", "removed_diverse"]

TWO = ("o_proj", "down_proj")                                  # prior WW layers
ALL = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_PROJ = re.compile(r"\.(%s)\.weight$" % "|".join(ALL))


def proj_kind(k):
    m = _PROJ.search(k)
    return m.group(1) if m else None


def auc_removed_below(removed_vals, intact_vals):
    n = 0; sc = 0.0
    for r in removed_vals:
        for i in intact_vals:
            n += 1
            sc += 1.0 if r < i else (0.5 if r == i else 0.0)
    return sc / n if n else float("nan")


def bootstrap_ci(score, rem_keys, int_keys, n_boot=2000):
    rem = np.array([score[k] for k in rem_keys])
    intc = np.array([score[k] for k in int_keys])
    rng = np.random.RandomState(0)
    a = []
    for _ in range(n_boot):
        rb = rem[rng.randint(0, len(rem), len(rem))]
        ib = intc[rng.randint(0, len(intc), len(intc))]
        a.append(auc_removed_below(rb.tolist(), ib.tolist()))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    t0 = time.time(); P = print
    P("[STEP 4] all-layer weight-cosine AUC")
    # base: load once, cast to fp16 to match the corpus's load_base_model (fp16)
    base = torch.load(BASE_BIN, map_location="cpu", weights_only=True)
    lin_keys = [k for k in base if proj_kind(k)]
    base_lin = {k: base[k].to(torch.float16) for k in lin_keys}
    del base
    P(f"  base linear keys: {len(lin_keys)}  ({(time.time()-t0):.0f}s)")

    clean_h = {s: safe_open(str(CK / f"clean_s{s}" / "model.safetensors"), "pt") for s in SEEDS}

    # accumulate cos for each variant against its leave-one-out clean mean reference,
    # for the two-layer subset and the all-layer set simultaneously.
    cos_two, cos_all = {}, {}
    for cls in CLASSES:
        for s in SEEDS:
            vh = safe_open(str(CK / f"{cls}_s{s}" / "model.safetensors"), "pt")
            others = [o for o in SEEDS if o != s]               # LOO clean seeds
            acc = {"two": [0.0, 0.0, 0.0], "all": [0.0, 0.0, 0.0]}  # dot,|d|^2,|r|^2
            for k in lin_keys:
                B = base_lin[k].to(torch.float32)
                V = vh.get_tensor(k).to(torch.float32)
                diff = V - B
                ref = torch.zeros_like(B)
                for o in others:
                    ref += clean_h[o].get_tensor(k).to(torch.float32) - B
                ref /= len(others)
                dot = float((diff * ref).sum())
                dn = float((diff * diff).sum())
                rn = float((ref * ref).sum())
                kind = proj_kind(k)
                acc["all"][0] += dot; acc["all"][1] += dn; acc["all"][2] += rn
                if kind in TWO:
                    acc["two"][0] += dot; acc["two"][1] += dn; acc["two"][2] += rn
            def cos(a):
                den = (a[1] ** 0.5) * (a[2] ** 0.5)
                return a[0] / den if den > 0 else 0.0
            cos_two[(cls, s)] = cos(acc["two"])
            cos_all[(cls, s)] = cos(acc["all"])
            P(f"  {cls}_s{s:<2} cos_two={cos_two[(cls,s)]:.4f}  cos_all={cos_all[(cls,s)]:.4f}"
              f"  ({(time.time()-t0):.0f}s)")

    rem_keys = [(c, s) for c in REMOVED for s in SEEDS]
    int_keys = [(c, s) for c in INTACT for s in SEEDS]

    def summarize(cosd, label):
        cls_mean = {c: float(np.mean([cosd[(c, s)] for s in SEEDS])) for c in CLASSES}
        intact_min = min(cls_mean[c] for c in INTACT)
        removed_max = max(cls_mean[c] for c in REMOVED)
        auc = auc_removed_below([cosd[k] for k in rem_keys], [cosd[k] for k in int_keys])
        lo, hi = bootstrap_ci(cosd, rem_keys, int_keys)
        # controls / detail
        ts_pass = bool(min(cosd[("trig_safe", s)] for s in SEEDS) > removed_max)
        floor = min(cosd[k] for k in int_keys)
        rd2 = cosd[("removed_diverse", 2)]; rd3 = cosd[("removed_diverse", 3)]
        return {"label": label, "cls_mean": cls_mean, "intact_min": intact_min,
                "removed_max": removed_max, "margin": intact_min - removed_max,
                "separates": bool(removed_max < intact_min), "auc": auc,
                "auc_ci": [lo, hi], "trig_safe_gate_pass": ts_pass,
                "removed_diverse_s2_cos": rd2, "removed_diverse_s2_caught": bool(rd2 < floor),
                "removed_diverse_s3_cos": rd3, "removed_diverse_s3_caught": bool(rd3 < floor)}

    two = summarize(cos_two, "two_layer_o_down")
    alll = summarize(cos_all, "all_linear_layers")

    P("\n" + "=" * 78)
    P("STEP 4 -- WEIGHT-COSINE AUC (mean leave-one-out clean reference)")
    P("=" * 78)
    for r in (two, alll):
        P(f"  {r['label']:<22} AUC {r['auc']:.3f}  CI[{r['auc_ci'][0]:.2f},{r['auc_ci'][1]:.2f}]"
          f"  margin {r['margin']:+.4f}  trig_safe {'PASS' if r['trig_safe_gate_pass'] else 'FAIL'}"
          f"  rd_s2 {'Y' if r['removed_diverse_s2_caught'] else 'n'}"
          f"  rd_s3 {'Y' if r['removed_diverse_s3_caught'] else 'n'}")
    prior = 0.9733
    d = alll["auc"] - two["auc"]
    rel = ("BEATS" if alll["auc"] > two["auc"] + 1e-9 else
           "MATCHES" if abs(d) < 1e-9 else "TRAILS")
    P(f"\n  reproduced two-layer AUC = {two['auc']:.3f} (prior run: {prior:.3f})")
    P(f"  all-layer AUC = {alll['auc']:.3f}  -> all-layer {rel} two-layer "
      f"(Delta {d:+.3f}); removal signal {'also lives in' if alll['auc']>=two['auc']-0.02 else 'is diluted by'} q,k,v,gate,up.")
    P("=" * 78)

    result = {"prior_two_layer_auc": prior, "two_layer": two, "all_layer": alll,
              "all_vs_two": rel, "delta": d, "cos_two": {f"{c}_s{s}": cos_two[(c, s)] for c in CLASSES for s in SEEDS},
              "cos_all": {f"{c}_s{s}": cos_all[(c, s)] for c in CLASSES for s in SEEDS},
              "runtime_min": (time.time() - t0) / 60}
    C.jdump(result, OUT / "exp1_weight_alllayer.json")
    P(f"  saved out/exp1_weight_alllayer.json  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
