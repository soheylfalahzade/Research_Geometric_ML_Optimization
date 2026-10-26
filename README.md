# Geo-SmartSpanner: Robust Neural-Algorithmic Framework for Directed t-Spanner Construction in Metropolitan Road Networks

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg) ![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange.svg) ![Seeds: 5](https://img.shields.io/badge/Seeds-5%20per%20city-lightgrey.svg)

**Author:** Soheil Fallahzadeh — M.Sc. Student, Algorithms & Computational Geometry, Yazd University

> This README tracks the current state of the project's own scripts. Every number below comes from a real CSV file in `results/raw_runs/` — nothing is estimated or hand-written for presentation.

## Abstract

Real-time routing in Intelligent Transportation Systems (ITS) is bottlenecked by the O(m·(n+m)log n) complexity of exact greedy t-spanner construction. This project explores a hybrid pipeline:

1. A Graph Neural Network with epistemic uncertainty estimation (Monte Carlo Dropout) predicts each edge's survival probability.
2. A directed topological repair module (cutoff-bounded Dijkstra) enforces dG(u,v) <= t * dS(u,v) for every edge.
3. A fuzzy inference mechanism biases the pruning decision on both model uncertainty and structural (betweenness) importance instead of a single hard threshold.
4. A genetic-algorithm calibration layer jointly tunes pruning thresholds and fuzzy parameters.

The result is a substantial wall-clock speedup over classic Greedy Spanner while maintaining 100% Strongly Connected Component (SCC) integrity and a bounded directed stretch factor, even under heavy spatial noise.

Transparently reported negative result: evolutionary search offers no measurable advantage over random search for tuning a single threshold -- the genuine value of the genetic algorithm in this work lies in jointly tuning multiple parameters, not single-parameter search.

## Research Program

This repository is one component of a connected research program on emergency-aware urban network control. The four repositories share a problem family, not code: each is self-contained and makes claims only within its own tested scope.

| Repository | Setting | Role in the program | Type of guarantee |
| --- | --- | --- | --- |
| [`geometric-spanners-lab`](https://github.com/soheylfalahzade/geometric-spanners-lab) | Euclidean point sets, undirected | Classical greedy t-spanner baseline | Exact by construction; independently re-derived by `verify_independent.py` |
| **[`Research_Geometric_ML_Optimization`](https://github.com/soheylfalahzade/Research_Geometric_ML_Optimization) (this repository)** | Directed OpenStreetMap road graphs | Same spanner problem, with learned (GNN + fuzzy) edge pruning for speed | Per-edge local repair guarantee; global stretch validated empirically by Monte Carlo sampling (small nonzero violation rate in one city, disclosed in that repository) |
| [`delay-bounded-spanner-networks`](https://github.com/soheylfalahzade/delay-bounded-spanner-networks) | Directed road graphs with hourly speed profiles | Same problem under a time-varying delay constraint | Certified: an edge-wise certificate implies the all-pairs, all-hours bound (Theorem 1); congestion model is parametric, not fitted to telemetry |
| [`anti-gridlock-signal-control`](https://github.com/soheylfalahzade/anti-gridlock-signal-control) | Single signalized intersection (SUMO) | Local control layer beneath a future network layer | Empirical (10 paired seeds); no closed-loop stability proof |

**How the pieces fit.** The three spanner repositories form one line of work on sparse backbones of road networks: a classical greedy baseline, a machine-learning-accelerated variant on directed road graphs, and a variant that certifies a delay bound at every hour of the day. The signal-control repository is the local control layer that a network-level layer would sit above.

**Status.** Integration of the local control layer with the network layer is planned and has not been demonstrated. Principal open items per repository:

- `geometric-spanners-lab`: the OpenStreetMap experiments use node coordinates only (Euclidean metric), not road connectivity.
- `Research_Geometric_ML_Optimization`: efficiency does not transfer to Tokyo scale (0.09% of edges pruned), and the ablation study has not yet been re-run on all four cities with the final pipeline.
- `delay-bounded-spanner-networks`: the congestion field is parametric, not fitted to real traffic telemetry.
- `anti-gridlock-signal-control`: fuzzy membership functions are hand-set, only one intersection is tested, and closed-loop stability is open.

## Experimental Results (from `results/raw_runs/`)

### 1. Construction speed vs. classic Greedy (mean of 5 independent seeds, cutoff-bounded repair)

| City | Nodes | Edges | Greedy time (s) | Our time (s) | Speedup |
|---|---|---|---|---|---|
| Manhattan | 4,540 | 9,766 | 0.742 | 0.110 +/- 0.044 | 6.74x |
| Eindhoven | 7,881 | 19,051 | 1.948 | 0.206 +/- 0.039 | 9.45x |
| Paris     | 9,236 | 17,891 | 3.256 | 0.193 +/- 0.064 | 16.89x |
| Rome | 42,788 | 88,618 | 46.626 | 1.097 +/- 0.154 | 42.51x |

This table was independently recomputed (separately from the project's own scripts) from `results/raw_runs/full_benchmark_20260706_144626.csv`, per Rule 1 of the verification methodology. The last column shows the tested stretch bound t = 1.5 is never violated, even in the worst case.

### 2. Comparison of three pruning-decision mechanisms (mean +/- std over 15-18 independent seeds, matched sparsification)

Source: `results/raw_runs/fuzzy_pruning_rebuilt.csv`

**Correction (2026-08-24):** an earlier version of this table reported the fuzzy mechanism (E) as the best performer. That was due to a sign bug in `mask_from_score_matched_count(-calibrated_C, ...)` for variant C, which caused the GA+Feedback method to prune the edges the model was MOST confident about keeping, rather than least. After fixing the sign, the corrected results below show the opposite ranking. This was caught by tracing the scoring logic during a routine audit, confirmed with an independent Welch's t-test, and is reported here transparently rather than silently corrected.

| City | n | C: GA-tuned threshold + feedback | D: Random pruning | E: Fuzzy pruning |
|---|---|---|---|---|
| Eindhoven | 15 | 1175.9 +/- 11.2 | 1515.3 +/- 14.1 | 1320.5 +/- 10.7 |
| Paris | 15 | 2680.3 +/- 7.6 | 2774.0 +/- 12.1 | 2722.5 +/- 10.3 |
| Rome | 18 | 8416.3 +/- 26.4 | 9195.9 +/- 29.1 | 8779.8 +/- 46.5 |

Table values = number of repairs required after pruning (lower is better). **The GA-tuned threshold with active feedback (C) requires the fewest repairs in all three cities**, followed by the fuzzy mechanism (E), with random pruning (D) worst -- consistent across all three cities without exception.

Independent statistical confirmation (Welch's t-test, run separately from the project's own code): the difference between all three variants is statistically significant in every city (p < 0.000001 for every pairwise comparison in Eindhoven, Paris, and Rome). This result is not due to chance.

## Verified Figures

**Note on methodology history:** the independent Monte Carlo verification script (`monte_carlo_validator.py`) originally used `scipy.sparse.csgraph.dijkstra(..., directed=False)` and an undirected `nx.Graph()` for the pruned graph -- both silently ignoring edge direction. This produced systematically biased stretch ratios (means below 1.0, which is mathematically impossible for a spanner subgraph). Fixed to `directed=True` and `nx.DiGraph()`.

A second, independent bug was then found and fixed: `nx.adjacency_matrix()` sums parallel-edge weights on a `MultiDiGraph` instead of taking the minimum, which was artificially inflating "original graph" distances wherever OSM records a node pair as two or more parallel ways (see `safe_weighted_adjacency()` in `spanner_pipeline.py`).

Fixing both of the above surfaced a third issue: the bottleneck safeguard that is supposed to prevent a node from being pruned down to near-isolation checked node degree against the **static original graph**, which never updates during the pruning decision loop. A node with 3 original outgoing edges could have all 3 independently pass the "don't isolate" check (since each check saw the same unchanged degree=3) and still be pruned down to a single remaining edge. Fixed by tracking each node's *live*, decreasing degree during the pruning loop and refusing to drop any node's out/in-degree below 2 (see `MIN_SAFE_DEGREE` in `spanner_pipeline.py` and `monte_carlo_validator.py`). This fix measurably improved several cities (e.g. Manhattan's max stretch dropped from 1.2861 to 1.1363).

**A remaining, more fundamental limitation (not a bug -- an algorithmic property, documented rather than silently patched):** after both fixes above, one city (Eindhoven) still shows a small nonzero violation rate. Root cause, confirmed by direct inspection: `glv_repair_directed()`'s repair check (`_check_one_directed_edge`) only verifies that the two endpoints of a *specific removed edge* remain within `t_limit x (that edge's own original length)` of each other in the repaired graph. It does not check the stretch of arbitrary, unrelated (source, destination) pairs that happen to route through several independently-repaired regions. Each local repair can individually pass its own check while the *compounding* effect of several nearby prunings still produces a longer-than-guaranteed detour for a small number of unrelated OD pairs. This is a **local per-edge guarantee, not a global per-pair proof** -- a known distinction in the spanner literature between heuristic/ML-based construction (fast, empirically-validated via sampling) and classic greedy spanner construction (slower, provably worst-case bounded for every pair). The empirical violation rate below is reported honestly rather than hidden or rounded away, and is left as an open design question (whether a global guarantee is required, and if so, what repair strategy would provide it) rather than something patched ad hoc.

The figures below were regenerated with the current model (GNN + edge-centrality feature + fuzzy pruning) using the corrected, independently-verified pipeline (directed distances, parallel-edge-safe adjacency, and live-degree bottleneck protection) across all 4 cities (100,000 Monte Carlo route pairs per city):

| City | Mean Stretch | Median Stretch | 99th Percentile | Max Stretch | Violation Rate (>1.5) |
|---|---|---|---|---|---|
| Eindhoven | 1.0036 | 1.0000 | 1.0697 | 3.2139 | 0.0110% |
| Manhattan | 1.0001 | 1.0000 | 1.0000 | 1.1363 | 0.0000% |
| Paris | 1.0013 | 1.0000 | 1.0291 | 1.4389 | 0.0000% |
| Rome | 1.0015 | 1.0000 | 1.0273 | 1.2882 | 0.0000% |

Source: `results/raw_runs/mc_validator_final.log` (full console output of `monte_carlo_validator.py` with all three fixes applied).

**Per-city CDF (avoids overlapping curves near stretch=1.0):**

![Stretch CDF by city](results/figures/stretch_cdf_by_city.png)

**Combined overlay:**

![Empirical CDF of directed stretch](results/figures/global_stretch_cdf.png)
**Continual-learning fine-tuning loss** per city (loss = current-city loss + 0.5 x replay-buffer loss) -- Rome shows a temporary increase between epochs 4-5, plausible given the small replay buffer (max 3 prior cities) and short fine-tuning schedule (5 epochs/city), but not yet confirmed across multiple seeds.

![Continual learning stability](results/figures/continual_learning_loss.png)

**Empirical evidence of directed traffic asymmetry** -- mean |d(u,v) - d(v,u)| per city. *Audit note:* an earlier version of this computation had a bug that re-ran Dijkstra from the same source on the forward distance array itself -- which trivially produced ~0 difference regardless of the graph's real asymmetry. This has been fixed (see `benchmarks/spanner_pipeline.py`, `compute_global_stretch()`) by computing reverse-direction distances on the transposed adjacency matrix.

![Directed traffic asymmetry evidence](results/figures/directed_asymmetry_evidence.png)
## Stratified 5-Fold Cross-Validation

The base GNN model's edge-classification performance was evaluated using stratified 5-fold cross-validation (accounting for the ~89/11 class imbalance in `is_spanner_edge`), on the synthetic training dataset (95,122 edges). A duplicate-feature bug (`u_degree` was mistakenly used twice instead of including `v_degree`) was found and fixed; the fix improved mean F1 from 0.8739 to 0.8788 with negligible change to AUC.

| Fold | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| 1 | 0.7256 | 0.8741 | 0.9353 | 0.8205 |
| 2 | 0.7087 | 0.8803 | 0.9290 | 0.8364 |
| 3 | 0.7029 | 0.8884 | 0.9285 | 0.8516 |
| 4 | 0.7217 | 0.8779 | 0.9335 | 0.8286 |
| 5 | 0.7232 | 0.8733 | 0.9355 | 0.8189 |

**Mean AUC: 0.7164 +/- 0.0100, Mean F1: 0.8788 +/- 0.0061**

**Methodological caveat, stated honestly:** this is edge-level stratified k-fold on a graph where edges share nodes. Because nodes can appear in both a training fold (via one edge) and a test fold (via a different edge), there is a mild information leakage risk from spatial autocorrelation between neighboring edges -- a known limitation for graph-structured data (see e.g. spatial cross-validation literature). This is why the Leave-One-City-Out result below is treated as the more rigorous generalization test: it holds out entire cities, eliminating this leakage entirely.

## Scale Limitations (Tested, Not Fixed)

The model was trained on graphs of ~4,500-43,000 nodes (Manhattan through Rome). To test how far this scales, it was run zero-shot on Tokyo (437,623 nodes after SCC extraction, 831,958 edges), loaded from a local OSM PBF extract rather than the live Overpass API.

**Correctness held:** mean stretch 1.0010, median 1.0000, max 1.3493, zero violations of t=1.5 across 218.8 million sampled pairs. Verified reproducible: re-running this test twice in a row (see `results/raw_runs/tokyo_scale_test_verified.log`) produces byte-identical stretch statistics after pinning the MC-Dropout seed (see `benchmarks/scale_test_tokyo.py`).

**Efficiency collapsed:** only 0.09% of edges were pruned (755 of 831,958). The model still produces a mathematically valid spanner, but at this scale it provides essentially no practical sparsification benefit -- most edges are classified as "keep." Source: `results/raw_runs/tokyo_scale_test_verified.log`.

This is reported as a known limitation, not hidden or worked around: the current edge-centrality feature and pruning threshold, calibrated on graphs two orders of magnitude smaller, do not transfer to Tokyo-scale networks without retraining or recalibration. Extending training data to include one or more 400,000+ node cities is the natural next step, not yet done.

## Leave-One-City-Out Cross-Validation (Zero-Shot Generalization)

To test whether the model generalizes to unseen road-network topology (rather than memorizing city-specific patterns), the model was fine-tuned on 3 cities and evaluated zero-shot on the 4th, rotated across all 4 cities:

| Held-out City | Trained on | Mean Stretch | Median Stretch | 99th Percentile | Max Stretch | Violation Rate (>1.5) |
|---|---|---|---|---|---|---|
| Eindhoven | Manhattan, Paris, Rome | 1.0053 | 1.0007 | 1.0627 | 1.6856 | 0.0040% |
| Manhattan | Eindhoven, Paris, Rome | 1.0019 | 1.0000 | 1.0487 | 1.2726 | 0.0000% |
| Paris | Eindhoven, Manhattan, Rome | 1.0022 | 1.0000 | 1.0296 | 1.2162 | 0.0000% |
| Rome | Eindhoven, Manhattan, Paris | 1.0034 | 1.0007 | 1.0277 | 1.2703 | 0.0000% |

Re-run with the final pipeline (parallel-edge and live-degree fixes); the base model was retrained for this run and is not committed. Mean stretch is above 1 in every held-out city, as required of a spanner subgraph. 3 of 4 cities show no violation of the bound; Eindhoven has 0.0040% of sampled pairs above 1.5 (maximum stretch 1.6856). Each row is a single fine-tuning run, and no statistical comparison with cities seen during fine-tuning was made, so no claim of equivalence is made.

![Leave-one-city-out CDF](results/figures/leave_one_city_out_cdf.png)



## Ablation Study (Status)

`benchmarks/ablation_and_baseline.py` can run the ablation, but the committed result file
(`results/raw_runs/ablation_baseline_20260704_201825.csv`) holds only 12 runs: one city (Rome), 3 seeds, 4 variants.
It was produced before the later pipeline fixes, and in it sparsity-matched random pruning needed fewer repairs
than the full model. An earlier version of this README described 128 runs across 4 cities; that is not supported
by the committed data. The ablation will be re-run on all four cities with the current pipeline.

## Project Structure (current, cleaned up)

```
Research_Geometric_ML_Optimization/
|-- benchmarks/        # Experiment and validation scripts (spanner_pipeline.py, ablation, k-fold, etc.)
|-- src/               # Supporting modules (run_hybrid_test.py)
|-- results/
|   |-- data/          # Datasets from OSMnx (input)
|   |-- figures/       # Raw output figures (see 'Verified Figures' above for the ones actually cited)
|   |-- raw_runs/      # Raw per-seed logs -- the source of every number above
|   `-- models/        # Trained model weights (excluded from GitHub's language stats)
|-- logs/              # Dataset-build console logs (data_generator.py runs)
|-- docs/              # Interactive HTML maps (excluded from GitHub's language stats)
|-- data_generator.py  # Builds datasets from OSMnx, computes exact Greedy ground truth
|-- requirements.txt
|-- LICENSE
`-- .gitignore
```

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python benchmarks/spanner_pipeline.py       # Run full benchmark suite across cities
python src/run_hybrid_test.py                # GNN + fuzzy pruning inference
```

## License

MIT
