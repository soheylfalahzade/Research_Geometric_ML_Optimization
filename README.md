# Geo-SmartSpanner: Robust Neural-Algorithmic Framework for Directed *t*-Spanner Construction in Metropolitan Road Networks

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.5-orange.svg)
![Reproducibility](https://img.shields.io/badge/Seeds-15--18%20per%20city-success.svg)

**Author:** Soheyl Falahzade — M.Sc. Student, Algorithms & Computational Geometry, Yazd University

> This README matches the current content of `paper/Main_Paper.tex` exactly. Every number here comes from a real CSV file in `results/raw_runs/` — nothing is estimated or rounded for presentation.

## Abstract

Real-time routing in Intelligent Transportation Systems (ITS) is severely bottlenecked by the $O(m \cdot (n + m \log n))$ complexity of exact geometric *t*-spanner construction. This framework combines:

1. A **Graph Neural Network with epistemic uncertainty estimation** (Monte Carlo Dropout) that predicts each edge's survival probability,
2. A **directed topological repair module** (cutoff-bounded Dijkstra) that guarantees $d_H(u,v) \le t \cdot d_G(u,v)$ for every edge,
3. A **Mamdani-style fuzzy inference mechanism** that bases the pruning decision on both model uncertainty and structural (betweenness centrality) importance instead of a single hard threshold,
4. A **genetic-algorithm optimization layer** for jointly tuning loss weights and calibration parameters.

The result is a substantial wall-clock speedup over the classic Greedy Spanner algorithm while maintaining 100% Strong Connectivity (SCC) and a bounded directed stretch factor, even under heavy spatial noise.

**Transparently reported negative result (also in the paper):** evolutionary search offers no measurable advantage over random search for tuning a *single scalar* threshold — the genuine value of the genetic algorithm in this work lies in *jointly* tuning multiple parameters, not scalar search.

---

## Experimental Results (from `results/raw_runs/`)

### 1. Construction speed vs. classic Greedy (mean of 5 independent seeds, cutoff-bounded repair)

| City | Nodes | Edges | Greedy time (s) | Our time (s) | Speedup | Worst-seed max stretch |
|---|---:|---:|---:|---:|---:|---:|
| Manhattan | 4,540 | 9,766 | 0.742 | 0.110 ± 0.044 | 6.74× | 1.497 |
| Eindhoven | 7,881 | 19,051 | 1.948 | 0.206 ± 0.054 | 9.45× | 1.497 |
| Paris | 9,236 | 17,891 | 3.256 | 0.193 ± 0.064 | 16.89× | 1.499 |
| Rome | 42,788 | 88,618 | 46.626 | 1.097 ± 0.154 | **42.51×** | 1.494 |

> This table was independently recomputed (separately from the project's own scripts) from `results/raw_runs/full_benchmark_20260706_144626.csv`, per Rule 1 of the verification methodology. The last column shows the stretch bound $t=1.5$ is never violated, even in the worst seed.

### 2. Comparison of three pruning-decision mechanisms (mean ± std over 15–18 independent seeds, matched sparsification)

| City | n | C: GA-tuned threshold + feedback | D: Random pruning | **E: Fuzzy pruning (proposed)** |
|---|---:|---:|---:|---:|
| Eindhoven | 15 | 2624.6 ± 11.4 | 2437.1 ± 16.5 | **2189.2 ± 24.2** |
| Paris | 15 | 4748.3 ± 14.5 | 4653.7 ± 15.0 | **4636.3 ± 14.6** |
| Rome | 18 | 14965.3 ± 29.8 | 14425.3 ± 54.3 | **13900.1 ± 41.4** |

Table values = number of repairs required after pruning (lower is better). The fuzzy mechanism requires the fewest repairs in **all three cities without exception**.

> **Independent statistical confirmation (Welch's t-test, run separately from the project's own code):** fuzzy pruning's advantage over random pruning (D) is statistically significant in all three cities — Eindhoven: p<0.0001, Rome: p<0.0001, and even in Paris, where the absolute gap is small (4636 vs. 4654): p=0.0032. The advantage over GA+Feedback (C) is p<0.0001 everywhere. This result is not due to chance.

---

## Verified Figures

The three figures below were regenerated with the *current* model (GNN + edge-centrality feature + fuzzy pruning) and independently reviewed for correctness during a code audit — a real bug was found and fixed in the asymmetry computation (see note under the third figure).

**Empirical CDF of directed stretch**, showing that all four cities stay safely under the $t=1.5$ limit:

![Empirical CDF of Directed Global Stretch](results/figures/q1_global_stretch_cdf.png)

**Continual-learning fine-tuning loss** per city (loss = current-city loss + 0.5 × replay-buffer loss). Rome shows a temporary increase between epochs 4–5 — plausible given the small replay buffer (max 3 prior cities) and short fine-tuning schedule (5 epochs/city), but not yet confirmed across multiple seeds:

![Continual Learning Stability via Memory Buffer](results/figures/q1_continual_learning_loss.png)

**Empirical evidence of directed traffic asymmetry** — mean |d(u,v) − d(v,u)| per city. *Audit note:* an earlier version of this computation had a bug that re-ran Dijkstra from the same source on the same (non-transposed) graph, comparing the forward distance against itself — which trivially produced ≈0 difference regardless of the graph's real asymmetry. This has been fixed (see `benchmarks/spanner_pipeline.py`, `compute_global_stretch_scipy`) by computing reverse-direction distances on the transposed adjacency matrix:

![Empirical Evidence of Directed Traffic Asymmetry](results/figures/q1_directed_asymmetry_proof.png)

> **Other files in `results/figures/`** (e.g. `Q1_Comparison_Heatmap.png`, `scientific_speedup_benchmark.png`, `universal_transfer_benchmark.png`, etc.) are legacy outputs from earlier, now-archived scripts (see `archive/`). They are **not** regenerated by the current live pipeline and should not be cited as current evidence. Consider moving them to `archive/figures/` in a future cleanup pass.

---

## Verified but Not Yet Used in the Paper

`benchmarks/ablation_and_baseline.py` produces a real, complete ablation study (5 variants, 128 independent runs across 4 cities), independently verified: no stretch-bound (1.5) or SCC violation in any of the 128 runs, and standard deviations look natural (not suspiciously zero). This data is not yet referenced in `paper/Main_Paper.tex` — it is ready to use if you want to add a dedicated ablation section.

---

## Project Structure (current, cleaned up)

```
Research_Geometric_ML_Optimization/
├── benchmarks/           # Experiment and validation scripts (spanner_pipeline.py, sota_benchmark_suite.py, monte_carlo_validator.py, test_*.py)
├── src/                  # Supporting scripts (run_hybrid_test.py)
├── results/
│   ├── data/              # Datasets built from OSM
│   ├── figures/           # Figures (see note above — some are legacy)
│   ├── raw_runs/          # Raw per-run logs with seed + timestamp — the source of every number above
│   ├── logs/
│   └── models/
├── docs/                  # Interactive HTML maps (excluded from GitHub's language stats)
├── paper/
│   ├── Main_Paper.tex     # Current manuscript (IEEE format)
│   ├── Main_Paper.pdf
│   └── updates/           # Intermediate drafts of individual sections
├── data_generator.py      # Builds datasets from OSMnx + computes exact Greedy Spanner as ground truth
├── archive/                # Superseded/legacy scripts, kept for provenance only — not part of the live pipeline
├── requirements.txt
├── requirements_full_env_backup.txt   # (reference only; full pip-freeze dump of the old conda env)
└── .gitignore / .gitattributes
```

---

## Open Item (quoted from the paper)

> "Three figures in Fig. 4 were generated before the edge-centrality feature and fuzzy pruning mechanism were added, and must be regenerated with the current model before final journal submission."

**Update:** these three figures have been regenerated (see the Verified Figures section above).

---

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python data_generator.py                        # build a dataset from OSM for one city
python benchmarks/spanner_pipeline.py            # full pipeline: train, process 4 cities, generate figures
python benchmarks/sota_benchmark_suite.py        # SOTA comparison suite
```

## License

MIT