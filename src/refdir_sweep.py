"""PART B -- reference-count + reference-type sweep (CPU, on saved vectors).

Eval set: the existing 25 gate-v2 variants. intact = clean/benign/trig_safe,
removed = removed/removed_diverse. References are built ONLY from clean-install
task vectors, leave-one-out (excluding the variant's own seed) so the detector
stays deployable.

Two reference types over N installs:
  flat-mean : reference = mean of N installs.  score(v) = cos(v, mean).
  svd       : top-k right singular directions (shared subspace) of the N installs.
              score(v) = in-subspace energy fraction = sum_m <v,q_m>^2 / ||v||^2
              (out-of-subspace fraction = 1 - that, also reported).
Both scores oriented so HIGH = intact (aligned with the install direction) and
LOW = removed (removal partially cancels the install). GUARDRAIL: only the
install-direction / subspace signal is used -- no cos_mean, no energy_frac.

Sweep N in {1,2,4,8,16}, k in {1,2,3,5,10} (svd only; k<=N). All references for a
variant exclude that variant's own seed; the N installs are the N lowest-indexed
remaining seeds (so flat-mean N=4 reproduces the earlier mean-LOO baseline).

The SVD subspace is computed via the NxN Gram matrix of the selected installs
(eigh), never materialising a full-D principal direction:
  right singular vec  q_m = (1/sqrt(lambda_m)) sum_i w_{m,i} install_i  (||q_m||=1)
  <v, q_m> = (1/sqrt(lambda_m)) sum_i w_{m,i} <v, install_i>
so every score reduces to precomputed <v,install_i>, <install_i,install_j>, ||v||^2.
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
EVAL_SEEDS = [0, 1, 2, 3, 4]
INTACT = ["clean", "benign", "trig_safe"]
REMOVED = ["removed", "removed_diverse"]
CLASSES = INTACT + REMOVED
N_GRID = [1, 2, 4, 8, 16]
K_GRID = [1, 2, 3, 5, 10]


def load_vec(path):
    return load_file(str(path))


def dot_dicts(a, b):
    s = 0.0
    for k in a:
        s += float((a[k].to(torch.float32) * b[k].to(torch.float32)).sum())
    return s


def norm2_dict(a):
    s = 0.0
    for k in a:
        x = a[k].to(torch.float32)
        s += float((x * x).sum())
    return s


def auc_removed_below_intact(removed_vals, intact_vals):
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
    P = print
    inst_paths = sorted(VEC.glob("install_s*.safetensors"),
                        key=lambda p: int(p.stem.split("_s")[1]))
    inst_seeds = [int(p.stem.split("_s")[1]) for p in inst_paths]
    if not inst_seeds:
        P(f"ERROR: no install vectors under {VEC}"); sys.exit(1)
    P("=" * 92)
    P("REFERENCE-COUNT / REFERENCE-TYPE SWEEP  (CPU; install-direction signal only)")
    P("=" * 92)
    P(f"  available clean installs: {len(inst_seeds)} (seeds {inst_seeds})")
    P(f"  eval set: 25 variants  intact={INTACT}  removed={REMOVED}")

    # ---- precompute dot products (hold installs in memory once) ----
    installs = {s: load_vec(VEC / f"install_s{s}.safetensors") for s in inst_seeds}
    # Gram of installs
    Gram = {i: {} for i in inst_seeds}
    for a in range(len(inst_seeds)):
        for b in range(a, len(inst_seeds)):
            i, j = inst_seeds[a], inst_seeds[b]
            d = dot_dicts(installs[i], installs[j])
            Gram[i][j] = d; Gram[j][i] = d
    # variant-install dots + variant norms (load each variant once)
    VI = {}; vnorm2 = {}
    for c in CLASSES:
        for s in EVAL_SEEDS:
            v = load_vec(VEC / f"{c}_s{s}.safetensors")
            VI[(c, s)] = {i: dot_dicts(v, installs[i]) for i in inst_seeds}
            vnorm2[(c, s)] = norm2_dict(v)
            del v
    del installs                      # free ~13GB

    def pool_for(seed, N):
        cand = [i for i in inst_seeds if i != seed]      # leave-one-out
        return cand[:N] if len(cand) >= N else None

    def flatmean_score(c, s, sel):
        N = len(sel)
        dot = sum(VI[(c, s)][i] for i in sel) / N
        refn2 = sum(Gram[i][j] for i in sel for j in sel) / (N * N)
        den = (vnorm2[(c, s)] * refn2) ** 0.5
        return dot / den if den > 0 else 0.0

    def svd_scores(c, s, sel, k):
        """in-subspace energy fraction for top-k shared directions over `sel`."""
        N = len(sel)
        Gs = np.array([[Gram[i][j] for j in sel] for i in sel], float)
        lam, W = np.linalg.eigh(Gs)                      # ascending
        order = np.argsort(lam)[::-1]
        lam = lam[order]; W = W[:, order]
        kk = min(k, N)
        p = np.array([VI[(c, s)][i] for i in sel], float)
        in_energy = 0.0
        for m in range(kk):
            if lam[m] <= 1e-6:
                continue
            proj = float(W[:, m] @ p) / (lam[m] ** 0.5)  # <v, q_m>
            in_energy += proj * proj
        in_frac = in_energy / vnorm2[(c, s)] if vnorm2[(c, s)] > 0 else 0.0
        return in_frac

    def evaluate(score_fn):
        """score_fn(c,s) -> float (high=intact). Returns metrics dict."""
        sc = {(c, s): score_fn(c, s) for c in CLASSES for s in EVAL_SEEDS}
        cls_mean = {c: float(np.mean([sc[(c, s)] for s in EVAL_SEEDS])) for c in CLASSES}
        intact_min = min(cls_mean[c] for c in INTACT)
        removed_max = max(cls_mean[c] for c in REMOVED)
        margin = intact_min - removed_max
        rem = [sc[(c, s)] for c in REMOVED for s in EVAL_SEEDS]
        intv = [sc[(c, s)] for c in INTACT for s in EVAL_SEEDS]
        auc = auc_removed_below_intact(rem, intv)
        intact_floor = min(intv)                         # lowest intact VARIANT
        s2 = sc[("removed_diverse", 2)] < intact_floor
        s3 = sc[("removed_diverse", 3)] < intact_floor
        ts_ctrl = cls_mean["trig_safe"] > removed_max    # trig_safe on intact side
        rd_ctrl = cls_mean["removed_diverse"] < intact_min  # removed_diverse on removed side
        return {"auc": auc, "margin": margin, "s2_below": bool(s2), "s3_below": bool(s3),
                "both_below": bool(s2 and s3), "ts_ctrl": bool(ts_ctrl),
                "rd_ctrl": bool(rd_ctrl), "controls_ok": bool(ts_ctrl and rd_ctrl),
                "intact_floor": intact_floor, "cls_mean": cls_mean}

    # ---- run the sweep ----
    rows = []
    # flat-mean: N only
    for N in N_GRID:
        if any(pool_for(s, N) is None for s in EVAL_SEEDS):
            continue
        m = evaluate(lambda c, s: flatmean_score(c, s, pool_for(s, N)))
        rows.append({"type": "flat-mean", "N": N, "k": None, **m})
    # svd: N x k (k<=N)
    for N in N_GRID:
        if any(pool_for(s, N) is None for s in EVAL_SEEDS):
            continue
        for k in K_GRID:
            if k > N:
                continue
            m = evaluate(lambda c, s, k=k, N=N: svd_scores(c, s, pool_for(s, N), k))
            rows.append({"type": "svd", "N": N, "k": k, **m})

    # baseline = flat-mean N=4 (the earlier mean-LOO)
    base = next((r for r in rows if r["type"] == "flat-mean" and r["N"] == 4), None)
    base_auc = base["auc"] if base else float("nan")

    P("\n  " + f"{'type':<10}{'N':>3}{'k':>4}{'AUC':>8}{'margin':>10}"
      f"{'s2<flr':>8}{'s3<flr':>8}{'both':>6}{'ctrls':>7}")
    for r in rows:
        kk = "" if r["k"] is None else r["k"]
        P("  " + f"{r['type']:<10}{r['N']:>3}{str(kk):>4}{r['auc']:>8.3f}"
          f"{r['margin']:>+10.4f}{('Y' if r['s2_below'] else '.'):>8}"
          f"{('Y' if r['s3_below'] else '.'):>8}{('YES' if r['both_below'] else 'no'):>6}"
          f"{('ok' if r['controls_ok'] else 'WOBBLE'):>7}")

    # ---- pick best & verdict ----
    cands = [r for r in rows if r["controls_ok"] and r["both_below"]]
    P("\n" + "-" * 92)
    if cands:
        best = max(cands, key=lambda r: (r["auc"], r["margin"]))
        small = (best["type"] == "flat-mean" and best["N"] <= 4) or \
                (best["type"] == "svd" and best["N"] <= 4 and (best["k"] or 0) <= 3)
        improves = best["auc"] >= base_auc - 1e-9
        kdesc = "" if best["k"] is None else f" k={best['k']}"
        if improves and small:
            verdict = "STRONGER"
            reason = (f"best={best['type']} N={best['N']}{kdesc} pulls removed_diverse "
                      f"s2 & s3 below the intact floor, AUC {best['auc']:.3f} "
                      f"(baseline {base_auc:.3f}), controls ok at modest N/k. "
                      f"Deployability firms up.")
        else:
            verdict = "PARTIAL"
            reason = (f"crossings fixed only at {best['type']} N={best['N']}{kdesc} "
                      f"(AUC {best['auc']:.3f}, baseline {base_auc:.3f}); needs large "
                      f"N/k. Trade-off noted.")
    else:
        # any config that fixed both crossings at all (ignoring controls)?
        any_fix = [r for r in rows if r["both_below"]]
        verdict = "CEILING"
        if any_fix:
            reason = (f"no config fixes BOTH crossings with controls intact "
                      f"(best AUC {max(r['auc'] for r in rows):.3f}, baseline "
                      f"{base_auc:.3f}); a control wobbles wherever the crossings clear.")
        else:
            reason = (f"no reference type / N / k pulls removed_diverse s2 & s3 below "
                      f"the intact floor (best AUC {max(r['auc'] for r in rows):.3f}, "
                      f"baseline {base_auc:.3f}). The diffuse-removal blind spot is "
                      f"real regardless of reference; evasion section confirmed.")
    P(f"  VERDICT: {verdict}")
    P(f"  {reason}")
    P("=" * 92)

    C.jdump({"verdict": verdict, "reason": reason, "baseline_auc": base_auc,
             "n_installs": len(inst_seeds), "install_seeds": inst_seeds,
             "configs": rows}, OUT / "gate_refdir_sweep.json")


if __name__ == "__main__":
    main()
