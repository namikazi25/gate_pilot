"""Robustness check for the sweep's STRONGER claim.

The deterministic sweep (lowest-N installs) found flat-mean N=2 gives AUC 1.000
with both removed_diverse crossings fixed, while N=4/8/16 stay at 0.973. That is
non-monotonic (more references -> worse), which smells like a small-sample
coincidence of WHICH installs were picked. This resamples the reference subset
RANDOMLY (per variant, leave-one-out) over many trials and reports the
distribution of AUC and the rate at which both crossings (removed_diverse s2 & s3
below the intact floor) actually clear. If N=2 only clears them sometimes while
larger N never does, the STRONGER result is noise and the true verdict is CEILING.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import numpy as np
import refdir_sweep as RS

VEC = RS.VEC
EVAL_SEEDS = RS.EVAL_SEEDS
INTACT, REMOVED, CLASSES = RS.INTACT, RS.REMOVED, RS.CLASSES
R = 200


def main():
    inst_seeds = [int(p.stem.split("_s")[1])
                  for p in sorted(VEC.glob("install_s*.safetensors"),
                                  key=lambda p: int(p.stem.split("_s")[1]))]
    print(f"robustness: {len(inst_seeds)} installs, {R} random reference subsets per (type,N)")
    installs = {s: RS.load_vec(VEC / f"install_s{s}.safetensors") for s in inst_seeds}
    Gram = {i: {} for i in inst_seeds}
    for a in range(len(inst_seeds)):
        for b in range(a, len(inst_seeds)):
            i, j = inst_seeds[a], inst_seeds[b]
            d = RS.dot_dicts(installs[i], installs[j]); Gram[i][j] = d; Gram[j][i] = d
    VI, vnorm2 = {}, {}
    for c in CLASSES:
        for s in EVAL_SEEDS:
            v = RS.load_vec(VEC / f"{c}_s{s}.safetensors")
            VI[(c, s)] = {i: RS.dot_dicts(v, installs[i]) for i in inst_seeds}
            vnorm2[(c, s)] = RS.norm2_dict(v); del v
    del installs

    def flat_score(c, s, sel):
        N = len(sel)
        dot = sum(VI[(c, s)][i] for i in sel) / N
        refn2 = sum(Gram[i][j] for i in sel for j in sel) / (N * N)
        den = (vnorm2[(c, s)] * refn2) ** 0.5
        return dot / den if den > 0 else 0.0

    def svd_score(c, s, sel, k):
        N = len(sel)
        Gs = np.array([[Gram[i][j] for j in sel] for i in sel], float)
        lam, W = np.linalg.eigh(Gs); order = np.argsort(lam)[::-1]
        lam, W = lam[order], W[:, order]
        p = np.array([VI[(c, s)][i] for i in sel], float)
        e = 0.0
        for m in range(min(k, N)):
            if lam[m] > 1e-6:
                pr = float(W[:, m] @ p) / (lam[m] ** 0.5); e += pr * pr
        return e / vnorm2[(c, s)] if vnorm2[(c, s)] > 0 else 0.0

    def trial_metrics(score_of):
        sc = {(c, s): score_of(c, s) for c in CLASSES for s in EVAL_SEEDS}
        cmean = {c: np.mean([sc[(c, s)] for s in EVAL_SEEDS]) for c in CLASSES}
        intact_min = min(cmean[c] for c in INTACT); removed_max = max(cmean[c] for c in REMOVED)
        rem = [sc[(c, s)] for c in REMOVED for s in EVAL_SEEDS]
        intv = [sc[(c, s)] for c in INTACT for s in EVAL_SEEDS]
        auc = RS.auc_removed_below_intact(rem, intv)
        floor = min(intv)
        both = (sc[("removed_diverse", 2)] < floor) and (sc[("removed_diverse", 3)] < floor)
        ctrl = (cmean["trig_safe"] > removed_max) and (cmean["removed_diverse"] < intact_min)
        return auc, both, ctrl

    print(f"\n  {'type':<10}{'N':>3}{'k':>3}{'AUC mean':>10}{'AUC sd':>9}"
          f"{'AUC=1 %':>9}{'both-fixed %':>14}{'ctrls-ok %':>12}")
    for kind in ["flat-mean", "svd-k1"]:
        for N in [2, 4, 8, 16]:
            aucs, boths, ctrls = [], [], []
            for t in range(R):
                rng = np.random.default_rng(1000 * N + t + (0 if kind == "flat-mean" else 7))
                def score_of(c, s, rng=rng, N=N, kind=kind):
                    pool = [i for i in inst_seeds if i != s]
                    sel = list(rng.choice(pool, size=N, replace=False))
                    return flat_score(c, s, sel) if kind == "flat-mean" else svd_score(c, s, sel, 1)
                a, b, ct = trial_metrics(score_of)
                aucs.append(a); boths.append(b); ctrls.append(ct)
            k = "-" if kind == "flat-mean" else 1
            print(f"  {kind:<10}{N:>3}{str(k):>3}{np.mean(aucs):>10.3f}{np.std(aucs):>9.3f}"
                  f"{100*np.mean([x==1.0 for x in aucs]):>9.1f}"
                  f"{100*np.mean(boths):>14.1f}{100*np.mean(ctrls):>12.1f}")
    print("\n  reading: if 'both-fixed %' is high only at N=2 and ~0 at N>=8, the")
    print("  STRONGER result is a small-sample coincidence -> true verdict CEILING.")


if __name__ == "__main__":
    main()
