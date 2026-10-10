#!/usr/bin/env python3
"""Independent check that the numbers reported in the write-up (paper/Main_Paper.tex)
match the raw CSV files in results/raw_runs/. Run from the repository root:

    python verify_reported_numbers.py

Re-derives (own pandas/scipy path, nothing imported from the pipeline):
  1. Table of repairs per pruning mechanism (mean +/- std, n per city).
  2. Speedup table (nodes, edges, speedup) and the speedup range quoted in the text.
  3. Ordering C < E < D in every seed, and Holm-corrected Wilcoxon p-values.
  4. Sparsification identical across variants for every seed.
Exit code 0 = every check passed.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAW = Path("results/raw_runs")
TEX = Path("paper/Main_Paper.tex")
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def main():
    tex = TEX.read_text(encoding="utf-8")
    f = pd.read_csv(RAW / "fuzzy_pruning_rebuilt.csv")
    key = {"C": "C_GA_Threshold_plus_Feedback", "D": "D_Random_Pruning(matched)", "E": "E_Fuzzy_Pruning"}

    for city in ["Eindhoven", "Paris", "Rome"]:
        sub = f[f.city == city]
        n = sub[sub.variant == key["C"]].shape[0]
        cells = []
        for k in "CDE":
            s = sub[sub.variant == key[k]].n_repairs
            cells.append(f"{s.mean():.1f} $\\pm$ {s.std():.1f}")
        for c in cells:
            check(f"{city}: '{c}' present in tex table", c in tex)
        check(f"{city}: n={n} stated in tex", re.search(rf"{city}\s*&\s*{n}\s*&", tex) is not None)

    piv = f.pivot_table(index=["city", "seed"], columns="variant", values="n_repairs")
    ok = bool(((piv[key["C"]] < piv[key["E"]]) & (piv[key["E"]] < piv[key["D"]])).all())
    check("ordering C < E < D holds in every seed of every city", ok)

    sp = f.pivot_table(index=["city", "seed"], columns="variant", values="sparsification_pct")
    check("sparsification identical across variants", float((sp.max(axis=1) - sp.min(axis=1)).max()) < 1e-9)

    pvals = []
    for city in ["Eindhoven", "Paris", "Rome"]:
        d = piv.loc[city]
        for a, b in [("C", "E"), ("E", "D"), ("C", "D")]:
            pvals.append(stats.wilcoxon(d[key[a]] - d[key[b]]).pvalue)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.zeros(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    check("Holm-adjusted p < 0.004 for all nine comparisons (text claim)", bool((adj < 0.004).all()), f"(max {adj.max():.2e})")

    b = pd.read_csv(RAW / "full_benchmark_20260706_144626.csv")
    for _, r in b.iterrows():
        nodes = f"{int(r.num_nodes):,}"
        sp_s = f"{r.speedup_mean:.2f}x"
        check(f"{r.city}: nodes {nodes} in tex", nodes in tex)
        check(f"{r.city}: speedup {sp_s} in tex", sp_s in tex)
    lo, hi = b.speedup_mean.min(), b.speedup_mean.max()
    check("speedup range 6.74--42.51 quoted in text", f"{lo:.2f}--{hi:.2f}" in tex, f"(data {lo:.2f}--{hi:.2f})")

    check("no stale 5.97--26.19 speedup range left", "5.97" not in tex and "26.19" not in tex)
    check("no stale claim 'without exception' for fuzzy", "without exception" not in tex)
    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
