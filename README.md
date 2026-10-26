# Geo-SmartSpanner: Learned Edge Pruning for Directed Road-Network Spanners

<p align="center">
  <a href="https://github.com/soheylfalahzade/Research_Geometric_ML_Optimization/actions/workflows/tests.yml"><img src="https://github.com/soheylfalahzade/Research_Geometric_ML_Optimization/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-Geometric-orange.svg" alt="PyTorch Geometric">
  <img src="https://img.shields.io/badge/README%20tables-generated%20from%20raw%20data-brightgreen.svg" alt="README tables generated from raw data">
</p>

<p align="center">
  <img src="results/figures/pipeline_animation.gif" alt="Animation: prune a directed road graph, restore edges whose detour is too long, verify the stretch on all ordered pairs" width="100%">
</p>

<p align="center"><sub>Mechanics of the pipeline on a small synthetic directed graph. The repair step is this repository's own <code>glv_repair_directed</code>; the pruning scores here are an edge-betweenness heuristic, not the trained network. Regenerate with <code>python make_animation.py</code>.</sub></p>

> Every number in the tables below is generated from the raw files in `results/raw_runs/` by `benchmarks/render_readme_tables.py`, and a test checks that `README.md` matches them. Nothing is estimated or typed by hand.

**Author:** Soheil Fallahzadeh — M.Sc. Student, Algorithms & Computational Geometry, Yazd University

## Contents

[Summary](#summary) | [Method](#method) | [Results](#results) | [Limitations](#limitations) | [Methodology notes](#methodology-notes) | [Reproducibility](#reproducibility) | [Research program](#research-program) | [Quickstart](#quickstart)

## Summary

Exact greedy $t$-spanner construction costs $O(m\,(n + m \log n))$, which is slow for metropolitan road networks. This repository studies a hybrid pipeline for **directed** road graphs:

1. a graph neural network with Monte Carlo Dropout predicts each edge's survival probability and its uncertainty;
2. a pruning decision turns the scores into a removal set, either by a calibrated threshold or by a fuzzy rule that combines model probability with edge betweenness;
3. a directed repair step restores every removed edge whose own endpoints would otherwise be further than $t$ times the edge length apart;
4. the result is verified by exhaustive or Monte Carlo stretch measurement on ordered pairs, and by strong-connectivity checks.

<!-- BEGIN:glance -->
| Quantity | Result |
| --- | --- |
| Construction speed-up over classic greedy | 6.7x to 42.5x on 4 cities (4,540 to 42,788 nodes) |
| Strong connectivity | preserved in every benchmark run |
| Pairs above the stretch bound, in-distribution (10⁵ per city) | none above 1.5 in 3 of 4 cities; Eindhoven: 0.0110% |
| Pairs above the stretch bound, leave-one-city-out | none above 1.5 in 3 of 4 cities; Eindhoven: 0.0040% |
| Pruning decision, fewest repairs | GA-tuned threshold with feedback (C), then fuzzy (E), then random (D), in every paired seed |
<!-- END:glance -->

The stretch guarantee of step 3 is **local**: it holds for each removed edge, not provably for every pair of nodes. Measured on $10^5$ pairs per city it held in three of four cities; see [Limitations](#limitations).

A negative result is also reported: evolutionary search shows no measurable advantage over random search when tuning a single threshold; the genetic algorithm is useful here for tuning several parameters jointly.

## Method

```mermaid
flowchart LR
    A["OSM road graph<br/>directed, largest SCC"] --> B["Edge features<br/>length, betweenness,<br/>degrees, PageRank"]
    B --> C["GNN edge classifier<br/>MC-Dropout uncertainty"]
    C --> D["Pruning decision<br/>threshold or fuzzy rule"]
    D --> E["Directed repair<br/>cutoff-bounded Dijkstra"]
    E --> F["Sparse directed spanner"]
    F --> G["Verification<br/>stretch on ordered pairs, SCC"]
```

Directed stretch of a pair is $d_S(u,v) / d_G(u,v)$ with both distances measured along edge direction; a spanner subgraph can never have stretch below 1. Nodes are protected from being pruned below a live in/out degree of 2 (`MIN_SAFE_DEGREE`).

The fuzzy decision uses two normalized inputs, calibrated model probability $p$ and edge betweenness $c$, with five rules: high $p$ keep; medium $p$ and high $c$ keep; medium $p$ and low $c$ prune; low $p$ and high $c$ keep; low $p$ and low $c$ prune. It is implemented with plain numpy (no fuzzy-logic library).

## Results

### Construction speed against classic greedy

<!-- BEGIN:speedup -->
| City | Nodes | Edges | Greedy time (s) | Our time (s) | Speedup | Edges removed | Worst stretch | Strongly connected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Manhattan | 4,540 | 9,766 | 0.742 | 0.110 ± 0.044 | **6.74x** | 0.50% | 1.4971 | yes |
| Eindhoven | 7,881 | 19,051 | 1.948 | 0.206 ± 0.054 | **9.45x** | 4.18% | 1.4970 | yes |
| Paris | 9,236 | 17,891 | 3.256 | 0.193 ± 0.064 | **16.89x** | 0.80% | 1.4989 | yes |
| Rome | 42,788 | 88,618 | 46.626 | 1.097 ± 0.154 | **42.51x** | 1.92% | 1.4941 | yes |
Source: `results/raw_runs/full_benchmark_20260706_144626.csv` (mean over 5 seeds; ± is the standard deviation of the construction time). Sparsification is the fraction of edges removed in the final graph; the speedup is obtained at a modest sparsification level. The classic greedy baseline is measured once per city.
<!-- END:speedup -->

### Comparison of three pruning-decision mechanisms

C prunes with a per-city genetic-algorithm-tuned threshold followed by one round of feedback fine-tuning on repaired edges; D prunes the same number of edges uniformly at random; E uses the fuzzy rule. All three remove exactly the same number of edges per seed, so the comparison isolates the quality of the decision.

<!-- BEGIN:pruning -->
| City | Seeds | C: GA-tuned threshold + feedback | D: random pruning | E: fuzzy pruning |
| --- | --- | --- | --- | --- |
| Eindhoven | 15 | **1175.9 ± 11.2** | 1515.3 ± 14.1 | 1320.5 ± 10.7 |
| Paris | 15 | **2680.3 ± 7.6** | 2774.0 ± 12.1 | 2722.5 ± 10.3 |
| Rome | 18 | **8416.3 ± 26.4** | 9195.9 ± 29.1 | 8779.8 ± 46.5 |

Table values are the number of repairs needed after pruning (lower is better). Sparsification is identical across the three variants for every seed.

| City | Comparison | Fewer repairs (first vs second) | Seeds in favour | Cohen's d_z (first minus second) | Wilcoxon p | Holm-adjusted p |
| --- | --- | --- | --- | --- | --- | --- |
| Eindhoven | C vs E | 11.0% | 15/15 | -14.0 | 6.5e-04 | 3.9e-03 |
| Eindhoven | E vs D | 12.9% | 15/15 | -13.9 | 6.5e-04 | 3.9e-03 |
| Eindhoven | C vs D | 22.4% | 15/15 | -25.2 | 6.5e-04 | 3.9e-03 |
| Paris | C vs E | 1.6% | 15/15 | -7.8 | 6.5e-04 | 3.9e-03 |
| Paris | E vs D | 1.9% | 15/15 | -6.5 | 6.5e-04 | 3.9e-03 |
| Paris | C vs D | 3.4% | 15/15 | -13.2 | 6.5e-04 | 3.9e-03 |
| Rome | C vs E | 4.1% | 18/18 | -7.6 | 2.0e-04 | 1.4e-03 |
| Rome | E vs D | 4.5% | 18/18 | -8.7 | 7.6e-06 | 6.9e-05 |
| Rome | C vs D | 8.5% | 18/18 | -35.0 | 7.6e-06 | 6.9e-05 |

Paired by seed, two-sided Wilcoxon signed-rank test, Holm correction over all 9 comparisons. Source: `results/raw_runs/fuzzy_pruning_rebuilt.csv`.
<!-- END:pruning -->

The calibrated threshold with feedback (C) needs the fewest repairs, the fuzzy rule (E) is in between and beats random pruning (D) in every seed, so the fuzzy rule carries real signal beyond the sparsity level, but it does not outperform calibrated thresholding. This ranking is the corrected one; see the correction note under [Methodology notes](#methodology-notes).

### Stretch verification (in-distribution)

<!-- BEGIN:stretch -->
| City | Mean stretch | Median | 99th percentile | Max | Violation rate (> 1.5) |
| --- | --- | --- | --- | --- | --- |
| Manhattan | 1.0001 | 1.0000 | 1.0000 | 1.1363 | 0.0000% |
| Eindhoven | 1.0036 | 1.0000 | 1.0697 | 3.2139 | 0.0110% |
| Paris | 1.0013 | 1.0000 | 1.0291 | 1.4389 | 0.0000% |
| Rome | 1.0015 | 1.0000 | 1.0273 | 1.2882 | 0.0000% |

10⁵ sampled directed route pairs per city. Source: `results/raw_runs/mc_validator_final.log`.
<!-- END:stretch -->

<p align="center">
  <img src="results/figures/stretch_cdf_by_city.png" alt="Empirical CDF of directed stretch per city" width="92%">
</p>

<p align="center">
  <img src="results/figures/global_stretch_cdf.png" alt="Combined empirical CDF of directed stretch" width="62%">
</p>

### Zero-shot generalization: leave-one-city-out

The model is fine-tuned on three cities and evaluated on the fourth, rotating over all four. Entire cities are held out, which avoids the leakage inherent in edge-level cross-validation, so this is the primary generalization test.

<!-- BEGIN:loco -->
| Held-out city | Fine-tuned on | Mean stretch | Median | 99th percentile | Max | Violation rate (> 1.5) |
| --- | --- | --- | --- | --- | --- | --- |
| Eindhoven | Manhattan, Paris, Rome | 1.0053 | 1.0007 | 1.0627 | 1.6856 | 0.0040% |
| Manhattan | Eindhoven, Paris, Rome | 1.0019 | 1.0000 | 1.0487 | 1.2726 | 0.0000% |
| Paris | Eindhoven, Manhattan, Rome | 1.0022 | 1.0000 | 1.0296 | 1.2162 | 0.0000% |
| Rome | Eindhoven, Manhattan, Paris | 1.0034 | 1.0007 | 1.0277 | 1.2703 | 0.0000% |

One fine-tuning run per fold, 10⁵ sampled pairs on the held-out city. Source: `results/raw_runs/leave_one_city_out_results.csv`.
<!-- END:loco -->

Mean stretch is above 1 in every held-out city, as it must be for a spanner subgraph. No statistical comparison with cities seen during fine-tuning was made, so no claim of equivalence is made.

<p align="center">
  <img src="results/figures/leave_one_city_out_cdf.png" alt="Leave-one-city-out stretch CDF" width="72%">
</p>

### Directed asymmetry and continual-learning stability

<p align="center">
  <img src="results/figures/directed_asymmetry_evidence.png" alt="Mean directed asymmetry per city" width="62%">
  <img src="results/figures/continual_learning_loss.png" alt="Continual-learning fine-tuning loss per city" width="62%">
</p>

The asymmetry plot shows mean $|d(u,v) - d(v,u)|$ per city, computed on the transposed adjacency matrix. The continual-learning loss is current-city loss plus $0.5\times$ replay-buffer loss; Rome shows a temporary increase between epochs 4 and 5, plausible given the small replay buffer and short schedule, but not confirmed across multiple seeds.

### Edge-classifier cross-validation (supplementary)

Stratified 5-fold cross-validation of the edge classifier on the synthetic training set (about 89/11 class imbalance). Because edges share nodes, neighbouring edges can appear in both a training and a test fold, so this estimate is optimistic and is **not** used to support any generalization claim.

<!-- BEGIN:kfold -->
| Fold | AUC | F1 | Precision | Recall |
| --- | --- | --- | --- | --- |
| 1 | 0.7260 | 0.8780 | 0.9346 | 0.8279 |
| 2 | 0.7135 | 0.8628 | 0.9343 | 0.8015 |
| 3 | 0.7024 | 0.8756 | 0.9307 | 0.8267 |
| 4 | 0.7133 | 0.8780 | 0.9311 | 0.8306 |
| 5 | 0.7271 | 0.8748 | 0.9356 | 0.8215 |
| **mean ± std** | **0.7164 ± 0.0103** | **0.8739 ± 0.0063** | **0.9333 ± 0.0022** | **0.8216 ± 0.0117** |

Source: `results/raw_runs/stratified_kfold_results.csv`.
<!-- END:kfold -->

## Limitations

- **The repair guarantee is local.** The repair check verifies, for each removed edge, that its own endpoints remain within $t$ times that edge's original length. It does not check arbitrary source-destination pairs, and independently repaired regions can compound into a longer detour for a few unrelated pairs. In Eindhoven this appears as a small nonzero violation rate (in-distribution and leave-one-city-out). Classic greedy construction bounds every pair provably. Whether a global guarantee is required, and which repair would provide it, is an open design question.
- **Scale.** Zero-shot on Tokyo (437,623 nodes after SCC extraction, 831,958 edges) the output was valid (mean stretch 1.0010, maximum 1.3493, no violation over 218.8 million sampled pairs, byte-identical on a repeated run, see `results/raw_runs/tokyo_scale_test_verified.log`), but only 0.09% of edges (755) were pruned, so there is essentially no sparsification benefit at that scale. The edge-centrality feature and threshold were calibrated on graphs about two orders of magnitude smaller; retraining or recalibration on 400,000+ node cities has not been done.
- **Ablation.** The committed ablation file holds 12 runs (Rome, 3 seeds, 4 variants) from before the later pipeline fixes, and in it sparsity-matched random pruning needed fewer repairs than the full model. It is therefore not reported as a result; it will be re-run on all four cities with the current pipeline.
- **Cities and seeds.** Four cities support a directional generalization claim, not a definitive one; leave-one-city-out has one run per fold.

## Methodology notes

<details>
<summary><b>Correction of the pruning-mechanism ranking (2026-08-24)</b></summary>

An earlier version of the pruning comparison reported the fuzzy mechanism (E) as the best performer. That was due to a sign bug in `mask_from_score_matched_count(-calibrated_C, ...)` for variant C, which caused the GA+Feedback method to prune the edges the model was MOST confident about keeping, rather than least. After fixing the sign, the corrected results show the opposite ranking. This was caught by tracing the scoring logic during an audit and is reported here rather than silently corrected.
</details>

<details>
<summary><b>Three measurement bugs found and fixed in the stretch validation</b></summary>

The independent Monte Carlo verification script (`monte_carlo_validator.py`) originally used `scipy.sparse.csgraph.dijkstra(..., directed=False)` and an undirected `nx.Graph()` for the pruned graph, both silently ignoring edge direction. This produced systematically biased stretch ratios (means below 1.0, which is mathematically impossible for a spanner subgraph). Fixed to `directed=True` and `nx.DiGraph()`.

A second, independent bug: `nx.adjacency_matrix()` sums parallel-edge weights on a `MultiDiGraph` instead of taking the minimum, which was artificially inflating "original graph" distances wherever OSM records a node pair as two or more parallel ways (see `safe_weighted_adjacency()` in `spanner_pipeline.py`).

A third issue: the bottleneck safeguard that prevents a node from being pruned down to near-isolation checked node degree against the **static original graph**, which never updates during the pruning loop. A node with 3 original outgoing edges could have all 3 independently pass the "don't isolate" check (each check saw the same unchanged degree of 3) and still be pruned down to a single remaining edge. Fixed by tracking each node's *live*, decreasing degree during the pruning loop and refusing to drop any node's out/in-degree below 2 (see `MIN_SAFE_DEGREE` in `spanner_pipeline.py` and `monte_carlo_validator.py`). This measurably improved several cities (for example Manhattan's maximum stretch dropped from 1.2861 to 1.1363).

All three have regression tests in `tests/test_core_algorithms.py`; each test fails when the corresponding bug is reintroduced.
</details>

<details>
<summary><b>Why the repair guarantee is local, in detail</b></summary>

`glv_repair_directed()`'s repair check (`_check_one_directed_edge`) only verifies that the two endpoints of a specific removed edge remain within `t_limit` times that edge's own original length of each other in the repaired graph. It does not check the stretch of arbitrary, unrelated (source, destination) pairs that happen to route through several independently repaired regions. Each local repair can individually pass its own check while the compounding effect of several nearby prunings still produces a longer-than-guaranteed detour for a small number of unrelated pairs. This is a local per-edge guarantee, not a global per-pair proof, which is the known distinction between heuristic or learned construction (fast, validated empirically by sampling) and classic greedy construction (slower, provably worst-case bounded for every pair).
</details>

<details>
<summary><b>Audit note on the directed-asymmetry computation</b></summary>

An earlier version re-ran Dijkstra from the same source on the forward distance array itself, which trivially produced a difference of about zero regardless of the graph's real asymmetry. It now computes reverse-direction distances on the transposed adjacency matrix (see `compute_global_stretch_scipy()` in `benchmarks/spanner_pipeline.py`); `tests/test_core_algorithms.py` guards against the regression.
</details>

## Reproducibility

| Check | How |
| --- | --- |
| Unit tests (directed repair, parallel edges, stretch measurement) | `pytest -q`; runs on every push without torch or network access |
| README tables match raw data | `python benchmarks/render_readme_tables.py --check` (also part of the test suite) |
| Regenerate the animation | `python make_animation.py` |
| Re-run the full benchmark | `python benchmarks/run_full_benchmark.py` (downloads the four city graphs; the speed-up table above uses the newest `full_benchmark_*.csv`) |
| Leave-one-city-out | `python benchmarks/leave_one_city_out.py` |

Raw result files and when they were produced (the live-degree safeguard and the parallel-edge fix were completed on 2026-08-29):

| Result | Source file | Run committed | Note |
| --- | --- | --- | --- |
| Stretch verification | `mc_validator_final.log` | 2026-08-29 | final pipeline |
| Leave-one-city-out | `leave_one_city_out_results.csv` | 2026-10-23 | final pipeline, one run per fold |
| Pruning-mechanism comparison | `fuzzy_pruning_rebuilt.csv` | 2026-08-24 | rebuilt after the sign correction |
| Speed benchmark | `full_benchmark_20260706_144626.csv` | 2026-08-16 (run 2026-07-06) | earlier than the final fixes |
| Edge-classifier cross-validation | `stratified_kfold_results.csv` | 2026-08-23 | supplementary |
| Ablation | `ablation_baseline_20260704_201825.csv` | 2026-07-05 | 12 runs only, see Limitations |
| GA versus random search | `adaptive_ga_feedback_20260705_021416.csv` | 2026-07-05 | basis of the negative result in the Summary |

Seeds and random-number settings are fixed in the scripts; the MC-Dropout seed is pinned for the scale test.

## Research program

This repository is one component of a connected research program on emergency-aware urban network control. The four repositories share a problem family, not code: each is self-contained and makes claims only within its own tested scope.

| Repository | Setting | Role in the program | Type of guarantee |
| --- | --- | --- | --- |
| [`geometric-spanners-lab`](https://github.com/soheylfalahzade/geometric-spanners-lab) | Euclidean point sets, undirected | Classical greedy $t$-spanner baseline | Exact by construction; independently re-derived by `verify_independent.py` |
| **[`Research_Geometric_ML_Optimization`](https://github.com/soheylfalahzade/Research_Geometric_ML_Optimization) (this repository)** | Directed OpenStreetMap road graphs | Same spanner problem, with learned (GNN + fuzzy) edge pruning for speed | Per-edge local repair guarantee; global stretch validated empirically by Monte Carlo sampling (small nonzero violation rate in one city, disclosed above) |
| [`delay-bounded-spanner-networks`](https://github.com/soheylfalahzade/delay-bounded-spanner-networks) | Directed road graphs with hourly speed profiles | Same problem under a time-varying delay constraint | Certified: an edge-wise certificate implies the all-pairs, all-hours bound (Theorem 1); congestion model is parametric, not fitted to telemetry |
| [`anti-gridlock-signal-control`](https://github.com/soheylfalahzade/anti-gridlock-signal-control) | Signalized intersection and a three-junction corridor (SUMO) | Local control layer beneath a future network layer | Empirical (10 paired seeds per evaluation); no closed-loop stability proof |

Integration of the local control layer with the network layer is planned and has not been demonstrated. Open items per repository: the OpenStreetMap experiments in the lab repository use node coordinates only; this repository does not transfer to Tokyo scale and its ablation is pending; the delay-bounded repository uses a parametric congestion field; the signal-control repository tests hand-set membership functions on at most three junctions.

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

pytest -q                                   # unit tests, no GPU or network needed
python benchmarks/spanner_pipeline.py       # full benchmark suite across cities
python src/run_hybrid_test.py               # GNN + fuzzy pruning inference
```

## Project structure

```
Research_Geometric_ML_Optimization/
|-- benchmarks/        # pipeline, experiments, validators, README table renderer
|-- src/               # supporting modules (run_hybrid_test.py)
|-- tests/             # unit tests (torch-free core) and README consistency check
|-- results/
|   |-- data/          # datasets from OSMnx (input)
|   |-- figures/       # figures cited above plus other generated plots
|   |-- raw_runs/      # raw per-seed logs and CSVs: the source of every number above
|   `-- models/        # trained model weights
|-- logs/              # dataset-build console logs
|-- docs/              # interactive HTML maps
|-- make_animation.py  # illustration of the pipeline mechanics
|-- data_generator.py  # builds datasets from OSMnx, computes exact greedy ground truth
`-- requirements.txt
```

## License and citation

MIT License. Citation metadata is in [`CITATION.cff`](CITATION.cff).
