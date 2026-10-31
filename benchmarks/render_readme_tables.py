"""Render the results tables of README.md from the raw files in results/raw_runs/.

Every table in the README sits between markers

    <!-- BEGIN:name -->  ...  <!-- END:name -->

and is generated here, so no number in those tables is typed by hand.

    python benchmarks/render_readme_tables.py            # rewrite the tables in README.md
    python benchmarks/render_readme_tables.py --check    # exit 1 if README.md differs from the data

Sources
    glance       summary of the four sources below
    speedup      latest results/raw_runs/full_benchmark_*.csv
    pruning      results/raw_runs/fuzzy_pruning_rebuilt.csv   (paired Wilcoxon, Holm correction)
    stretch      results/raw_runs/mc_validator_final.log
    loco         results/raw_runs/leave_one_city_out_results.csv
    kfold        results/raw_runs/stratified_kfold_results.csv
    ablation     newest results/raw_runs/ablation_baseline_*.csv per city that has an n_removed column
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "results", "raw_runs")
README = os.path.join(ROOT, "README.md")
CITY_ORDER = ["Manhattan", "Eindhoven", "Paris", "Rome"]
VARIANTS = {"C": "C_GA_Threshold_plus_Feedback", "D": "D_Random_Pruning(matched)", "E": "E_Fuzzy_Pruning"}


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def table_speedup():
    path = sorted(glob.glob(os.path.join(RAW, "full_benchmark_*.csv")))[-1]
    df = pd.read_csv(path).set_index("city").loc[CITY_ORDER].reset_index()
    rows = []
    for _, r in df.iterrows():
        rows.append([r.city, f"{int(r.num_nodes):,}", f"{int(r.num_edges):,}", f"{r.greedy_time_sec:.3f}",
                     f"{r.gnn_time_mean_sec:.3f} \u00b1 {r.gnn_time_std_sec:.3f}", f"**{r.speedup_mean:.2f}x**",
                     f"{r.gnn_sparsification_mean_pct:.2f}%", f"{r.gnn_max_stretch_worst:.4f}",
                     "yes" if r.gnn_scc_all_connected else "no"])
    over = df[df.gnn_max_stretch_worst > 1.5]
    stretch_note = ("" if over.empty else
                    f" The worst stretch is the maximum over sampled source nodes and exceeds 1.5 in {len(over)} of {len(df)} cities "
                    f"({', '.join(f'{r.city}: {r.gnn_max_stretch_worst:.2f}' for _, r in over.iterrows())}); the repair guarantee is local, "
                    f"see Limitations.")
    note = (f"\nSource: `results/raw_runs/{os.path.basename(path)}` (mean over {int(df.gnn_n_seeds.iloc[0])} seeds; "
            f"\u00b1 is the standard deviation of the construction time). "
            f"Sparsification is the fraction of edges removed in the final graph; the speedup is obtained at a modest "
            f"sparsification level. The classic greedy baseline is measured once per city." + stretch_note)
    head = ["City", "Nodes", "Edges", "Greedy time (s)", "Our time (s)", "Speedup", "Edges removed",
            "Worst stretch", "Strongly connected"]
    return md(head, rows) + note


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.zeros(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def table_pruning():
    df = pd.read_csv(os.path.join(RAW, "fuzzy_pruning_rebuilt.csv"))
    piv = df.pivot_table(index=["city", "seed"], columns="variant", values="n_repairs")
    cities = [c for c in ["Eindhoven", "Paris", "Rome"] if c in piv.index.get_level_values(0)]
    rows, tests = [], []
    for c in cities:
        sub = piv.loc[c]
        cell = lambda v: f"{sub[VARIANTS[v]].mean():.1f} \u00b1 {sub[VARIANTS[v]].std():.1f}"  # noqa: E731
        rows.append([c, len(sub), f"**{cell('C')}**", cell("D"), cell("E")])
        for a, b in (("C", "E"), ("E", "D"), ("C", "D")):
            diff = sub[VARIANTS[a]] - sub[VARIANTS[b]]
            tests.append((c, a, b, 100 * (sub[VARIANTS[b]].mean() - sub[VARIANTS[a]].mean()) / sub[VARIANTS[b]].mean(),
                          stats.wilcoxon(diff).pvalue, float(diff.mean() / diff.std(ddof=1)),
                          int((diff < 0).sum()), len(diff)))
    adj = holm([t[4] for t in tests])
    t_rows = [[c, f"{a} vs {b}", f"{red:.1f}%", f"{w}/{n}", f"{d:.1f}", f"{p:.1e}", f"{q:.1e}"]
              for (c, a, b, red, p, d, w, n), q in zip(tests, adj)]
    main = md(["City", "Seeds", "C: GA-tuned threshold + feedback", "D: random pruning", "E: fuzzy pruning"], rows)
    sec = md(["City", "Comparison", "Fewer repairs (first vs second)", "Seeds in favour", "Cohen's d_z (first minus second)",
              "Wilcoxon p", "Holm-adjusted p"], t_rows)
    return (main + "\n\nTable values are the number of repairs needed after pruning (lower is better). Sparsification is "
            "identical across the three variants for every seed.\n\n" + sec +
            f"\n\nPaired by seed, two-sided Wilcoxon signed-rank test, Holm correction over all {len(tests)} comparisons. "
            "Source: `results/raw_runs/fuzzy_pruning_rebuilt.csv`.")


def table_stretch():
    text = open(os.path.join(RAW, "mc_validator_final.log"), encoding="utf-8").read()
    block = text.split("MONTE CARLO STRETCH VALIDATION METRICS")[-1]
    rows = []
    for line in block.splitlines():
        m = re.match(r"\s*([A-Za-z]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+%)\s*$", line)
        if m:
            rows.append(list(m.groups()))
    rows.sort(key=lambda r: CITY_ORDER.index(r[0]))
    assert len(rows) == 4, "could not parse the validator log"
    return (md(["City", "Mean stretch", "Median", "99th percentile", "Max", "Violation rate (> 1.5)"], rows) +
            "\n\n10\u2075 sampled directed route pairs per city. Source: `results/raw_runs/mc_validator_final.log`.")


def table_loco():
    df = pd.read_csv(os.path.join(RAW, "leave_one_city_out_results.csv"), dtype=str)
    rows = [[r["Held-out City"], r["Trained on"], r["Mean Stretch"], r["Median Stretch"], r["99th Percentile"],
             r["Max Stretch"], r["Violation Rate (>1.5)"]] for _, r in df.iterrows()]
    return (md(["Held-out city", "Fine-tuned on", "Mean stretch", "Median", "99th percentile", "Max",
                "Violation rate (> 1.5)"], rows) +
            "\n\nOne fine-tuning run per fold, 10\u2075 sampled pairs on the held-out city. "
            "Source: `results/raw_runs/leave_one_city_out_results.csv`.")


def table_kfold():
    df = pd.read_csv(os.path.join(RAW, "stratified_kfold_results.csv"))
    rows = [[int(r.fold), f"{r.auc:.4f}", f"{r.f1:.4f}", f"{r.precision:.4f}", f"{r.recall:.4f}"] for _, r in df.iterrows()]
    rows.append(["**mean \u00b1 std**", f"**{df.auc.mean():.4f} \u00b1 {df.auc.std():.4f}**",
                 f"**{df.f1.mean():.4f} \u00b1 {df.f1.std():.4f}**", f"**{df.precision.mean():.4f} \u00b1 {df.precision.std():.4f}**",
                 f"**{df.recall.mean():.4f} \u00b1 {df.recall.std():.4f}**"])
    return md(["Fold", "AUC", "F1", "Precision", "Recall"], rows) + "\n\nSource: `results/raw_runs/stratified_kfold_results.csv`."


def table_glance():
    bench = pd.read_csv(sorted(glob.glob(os.path.join(RAW, "full_benchmark_*.csv")))[-1])
    mc = open(os.path.join(RAW, "mc_validator_final.log"), encoding="utf-8").read().split("MONTE CARLO STRETCH VALIDATION METRICS")[-1]
    mc_rows = [m.groups() for m in (re.match(r"\s*([A-Za-z]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s*$", l) for l in mc.splitlines()) if m]
    loco = pd.read_csv(os.path.join(RAW, "leave_one_city_out_results.csv"), dtype=str)
    prune = pd.read_csv(os.path.join(RAW, "fuzzy_pruning_rebuilt.csv")).pivot_table(index=["city", "seed"], columns="variant", values="n_repairs")
    ordered = bool(((prune[VARIANTS["C"]] < prune[VARIANTS["E"]]) & (prune[VARIANTS["E"]] < prune[VARIANTS["D"]])).all())

    def viol(rows, city_i, rate_i):
        bad = [(r[city_i], r[rate_i]) for r in rows if float(str(r[rate_i]).rstrip("%")) > 0]
        return "none above 1.5 in " + str(len(rows) - len(bad)) + " of " + str(len(rows)) + " cities" + \
            ("; " + ", ".join(f"{c}: {v}%" if not str(v).endswith("%") else f"{c}: {v}" for c, v in bad) if bad else "")
    rows = [
        ["Construction speed-up over classic greedy", f"{bench.speedup_mean.min():.1f}x to {bench.speedup_mean.max():.1f}x on {len(bench)} cities "
                                                      f"({int(bench.num_nodes.min()):,} to {int(bench.num_nodes.max()):,} nodes)"],
        ["Strong connectivity", "preserved in every benchmark run" if bench.gnn_scc_all_connected.all() else "not preserved in every run"],
        ["Pairs above the stretch bound, in-distribution (10\u2075 per city)", viol(mc_rows, 0, 5)],
        ["Pairs above the stretch bound, leave-one-city-out", viol([(r["Held-out City"], r["Violation Rate (>1.5)"]) for _, r in loco.iterrows()], 0, 1)],
        ["Pruning decision, fewest repairs", "GA-tuned threshold with feedback (C), then fuzzy (E), then random (D), in every paired seed" if ordered
         else "ordering C < E < D does not hold in every seed"],
    ]
    return md(["Quantity", "Result"], rows)

ABL_LABELS = [("Full_Proposed", "Full model (MC-Dropout + GA threshold + centrality)"),
              ("Ablation_No_MC_Dropout", "No MC-Dropout"),
              ("Ablation_Non_GA_Threshold", "Untuned threshold 0.50"),
              ("Baseline_Random_Pruning", "Random pruning, same number of removed edges")]


def load_ablation():
    """Newest file per city that records n_removed (earlier runs have no such column: unmatched baseline, retrained weights)."""
    newest = {}
    for path in sorted(glob.glob(os.path.join(RAW, "ablation_baseline_*.csv"))):
        df = pd.read_csv(path)
        if "n_removed" not in df.columns:
            continue
        df["variant"] = df.variant.str.split("(").str[0].str.strip()
        newest[df.city.iloc[0]] = (path, df)  # sorted by name, so the newest file per city wins
    assert newest, "no ablation file with n_removed"
    return newest


def table_ablation():
    newest = load_ablation()
    rows, pairs, per_city = [], [], []
    for city in [c for c in CITY_ORDER if c in newest]:
        df = newest[city][1]
        for key, label in ABL_LABELS:
            sub = df[df.variant == key]
            rate = sub.n_repairs / sub.n_removed
            rows.append([city, label, len(sub), f"{sub.removed_pct.mean():.2f} \u00b1 {sub.removed_pct.std():.2f}",
                         f"{sub.n_repairs.mean():.1f} \u00b1 {sub.n_repairs.std():.1f}", f"{rate.mean():.3f} \u00b1 {rate.std():.3f}"])
        piv = df.pivot_table(index="seed", columns="variant", values="n_repairs")
        rem = df.pivot_table(index="seed", columns="variant", values="n_removed")
        assert (rem["Full_Proposed"] == rem["Baseline_Random_Pruning"]).all(), f"{city}: random baseline is not matched on removed edges"
        d = piv["Baseline_Random_Pruning"] - piv["Full_Proposed"]
        pairs.append(((d > 0).sum(), len(d)))
        per_city.append(f"{city} {int((d > 0).sum())}/{len(d)} (mean difference {d.mean():.1f})")
    k, n = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    files = ", ".join(f"`{os.path.basename(newest[c][0])}`" for c in CITY_ORDER if c in newest)
    return (md(["City", "Variant", "Seeds", "Edges removed before repair (% of edges)", "Repairs", "Repairs per removed edge"], rows) +
            "\n\nValues are mean \u00b1 standard deviation over seeds; repairs are counted after pruning (lower is better). "
            f"Random pruning removes exactly as many edges as the full model removed in the same seed, under the same degree safeguard. "
            f"The full model needed fewer repairs than this baseline in {k} of {n} city-seed pairs (by city: " + "; ".join(per_city) +
            "; difference = random minus full). The other two variants remove more edges than the full model, so their repair counts are "
            "not directly comparable; the last column is a rate, not a test. "
            "No significance test is reported: there are 5 seeds per city, and the seeds of one city share the graph and the trained model. "
            "Sources: " + files + ".")


RENDERERS = {"glance": table_glance, "speedup": table_speedup, "pruning": table_pruning, "stretch": table_stretch,
             "loco": table_loco, "kfold": table_kfold, "ablation": table_ablation}


def render(readme_text):
    for name, fn in RENDERERS.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->)\n.*?(<!-- END:{name} -->)", re.S)
        if not pat.search(readme_text):
            sys.exit(f"README.md has no marker block for '{name}'")
        readme_text = pat.sub(lambda m: m.group(1) + "\n" + fn() + "\n" + m.group(2), readme_text)
    return readme_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    current = open(README, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in current else "\n"
    rendered = render(current.replace("\r\n", "\n")).replace("\n", nl)
    if args.check:
        if rendered != current:
            print("README.md tables differ from the raw data; run: python benchmarks/render_readme_tables.py")
            sys.exit(1)
        print("README.md tables match the raw data.")
        return
    open(README, "w", encoding="utf-8", newline="").write(rendered)
    print("README.md tables rendered from results/raw_runs/")


if __name__ == "__main__":
    main()
