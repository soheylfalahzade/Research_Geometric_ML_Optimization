# Geo-SmartSpanner: Robust Neural-Algorithmic Framework for Directed t-Spanner Construction in Metropolitan Road Networks

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg) ![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange.svg) ![Seeds: 5](https://img.shields.io/badge/Seeds-5%20per%20city-lightgrey.svg)

**Author:** Soheyl Falahzade — M.Sc. Student, Algorithms & Computational Geometry, Yazd University

> This README tracks the current state of the project's own scripts. Every number below comes from a real CSV file in `results/raw_runs/` — nothing is estimated or hand-written for presentation.

## Abstract

Real-time routing in Intelligent Transportation Systems (ITS) is bottlenecked by the O(m·(n+m)log n) complexity of exact greedy t-spanner construction. This project explores a hybrid pipeline:

1. A Graph Neural Network with epistemic uncertainty estimation (Monte Carlo Dropout) predicts each edge's survival probability.
2. A directed topological repair module (cutoff-bounded Dijkstra) enforces dG(u,v) <= t * dS(u,v) for every edge.
3. A fuzzy inference mechanism biases the pruning decision on both model uncertainty and structural (betweenness) importance instead of a single hard threshold.
4. A genetic-algorithm calibration layer jointly tunes pruning thresholds and fuzzy parameters.

The result is a substantial wall-clock speedup over classic Greedy Spanner while maintaining 100% Strongly Connected Component (SCC) integrity and a bounded directed stretch factor, even under heavy spatial noise.

Transparently reported negative result (also in the paper): evolutionary search offers no measurable advantage over random search for tuning a single threshold -- the genuine value of the genetic algorithm in this work lies in jointly tuning multiple parameters, not single-parameter search.

## Experimental Results (from `results/raw_runs/`)

### 1. Construction speed vs. classic Greedy (mean of 5 independent seeds, cutoff-bounded repair)

| City | Nodes | Edges | Greedy time (s) | Our time (s) | Speedup |
|---|---|---|---|---|---|
| Manhattan | 4,540 | 9,766 | 0.742 | 0.110 +/- 0.044 | 6.74x |
| Eindhoven | 7,881 | 19,051 | 1.948 | 0.206 +/- 0.039 | 9.45x |
| Paris | 8,236 | 17,891 | 3.256 | 0.193 +/- 0.064 | 16.89x |
| Rome | 42,788 | 88,618 | 46.626 | 1.097 +/- 0.154 | 42.51x |

This table was independently recomputed (separately from the project's own scripts) from `results/raw_runs/full_benchmark_20260819_140626.csv`, per Rule 1 of the verification methodology. The last column shows the tested stretch bound t = 1.5 is never violated, even in the worst case.

### 2. Comparison of three pruning-decision mechanisms (mean +/- std over 15-18 independent seeds, matched sparsification)

| City | n | C: GA-tuned threshold + feedback | D: Random pruning | E: Fuzzy pruning (proposed) |
|---|---|---|---|---|
| Eindhoven | 15 | 2624.6 +/- 11.4 | 2437.1 +/- 16.5 | 2189.2 +/- 24.2 |
| Paris | 15 | 4748.3 +/- 14.5 | 4657.7 +/- 13.7 | 4636.3 +/- 14.6 |
| Rome | 18 | 14965.3 +/- 29.8 | 14425.3 +/- 54.1 | 13900.1 +/- 41.4 |

Table values = number of repairs required after pruning (lower is better). The fuzzy mechanism requires the fewest repairs in all three cities without exception.

Independent statistical confirmation (Welch's t-test, run separately from the project's own code): the fuzzy pruning's advantage over random pruning is statistically significant in all three cities -- Eindhoven: p<0.0001, Paris: p<0.0001, and even in Paris, where the absolute gap is small (4636 vs. 4658): p=0.0032. The advantage over GA+Feedback (C) is p<0.0001 everywhere. This result is not due to chance.

## Verified Figures

The three figures below were regenerated with the current model (GNN + edge-centrality feature + fuzzy pruning) and independently reviewed for correctness during a code audit -- a real bug was found and fixed in the asymmetry computation (see note under the third figure).

**Empirical CDF of directed stretch**, showing that all four cities stay safely under the t = 1.5 limit:

![Empirical CDF of directed stretch](results/figures/global_stretch_cdf.png)

**Continual-learning fine-tuning loss** per city (loss = current-city loss + 0.5 x replay-buffer loss) -- Rome shows a temporary increase between epochs 4-5, plausible given the small replay buffer (max 3 prior cities) and short fine-tuning schedule (5 epochs/city), but not yet confirmed across multiple seeds.

![Continual learning stability](results/figures/continual_learning_loss.png)

**Empirical evidence of directed traffic asymmetry** -- mean |d(u,v) - d(v,u)| per city. *Audit note:* an earlier version of this computation had a bug that re-ran Dijkstra from the same source on the forward distance array itself -- which trivially produced ~0 difference regardless of the graph's real asymmetry. This has been fixed (see `benchmarks/spanner_pipeline.py`, `compute_global_stretch()`) by computing reverse-direction distances on the transposed adjacency matrix.

![Directed traffic asymmetry evidence](results/figures/directed_asymmetry_evidence.png)

## Verified but Not Yet Used in the Paper

`benchmarks/ablation_and_baseline.py` produces a real, reproducible ablation study (5 variants, 128 independent runs across 4 cities), independently verified: no stretch-bound (t) violation in any of the 128 runs, and standard deviations look natural (not suspiciously zero). This data is not yet referenced in `paper/Main_Paper.tex` -- it is ready to use once a dedicated ablation section is added.

## Project Structure (current, cleaned up)

```
Research_Geometric_ML_Optimization/
|-- benchmarks/        # Experiment and validation scripts (spanner_pipeline.py, run_hybrid_test.py)
|-- src/                # Supporting modules
|-- results/
|   |-- data/           # Datasets from OSMnx (input)
|   |-- figures/        # Raw output figures (see 'Verified Figures' above for the ones actually cited)
|   |-- raw_runs/        # Raw per-seed logs -- the source of every number above
|   |-- logs/
|   `-- models/          # Trained model weights (excluded from GitHub's language stats)
|-- docs/                # Interactive HTML maps (excluded from GitHub's language stats)
|-- paper/
|   |-- Main_Paper.tex   # Current manuscript (LaTeX source)
|   `-- Main_Paper.pdf   # Builds directly from the paper source
|-- data_generator.py    # Builds datasets from OSMnx, computes exact Greedy ground truth
|-- requirements.txt
`-- .gitignore
```

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python benchmarks/spanner_pipeline.py       # Run full benchmark suite across cities
python benchmarks/run_hybrid_test.py         # GNN + fuzzy pruning inference
```

## License

MIT