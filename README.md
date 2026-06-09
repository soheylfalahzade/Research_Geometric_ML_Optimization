# Geospatial Graph Optimizer: AI-Driven t-Spanner Construction

An AI-driven optimization framework for real-time traffic management. We replace computationally expensive greedy spanner constructions with a Machine Learning classifier to prune redundant edges in massive geospatial road networks (12k+ nodes).

---

## 🔬 Scientific Methodology
This framework bridges Computational Geometry with Predictive Modeling:
1. **Topological Feature Engineering:** Maps local (node degree, centrality) and global (PageRank) network features to edge-pruning decisions.
2. **Metaheuristic Optimization:** Employs a **Genetic Algorithm (GA)** to find optimal hyperparameters (depth, estimators) for our Random Forest classifier, maximizing the trade-off between pruning density and topological connectivity.
3. **Safety-Constrained Inference:** Optimized to maintain connectivity while achieving high-speed routing.

## 📊 Benchmark Comparison (Computational Efficiency)
| Algorithm | Best Fitness Score | Optimization Budget |
| :--- | :--- | :--- |
| **Genetic Algorithm (GA)** | **0.7868** | 24 Evaluations |
| Particle Swarm (PSO) | 0.7836 | 24 Evaluations |
| Differential Evolution | 0.7814 | 24 Evaluations |
| Random Search (RS) | 0.7791 | 24 Evaluations |

---

## 🚀 Industrial & Academic KPIs
* **Performance:** 14x acceleration in shortest-path queries (under 2ms execution time).
* **Network Sparsification:** 29.6% reduction in edge density, optimizing storage and query throughput.
* **Generalization:** Demonstrated robustness on real-world GIS data from OpenStreetMap (Eindhoven topology).

---

## 🛠️ Repository Features
* `data_generator.py`: Generates ground truth labels using the rigorous Greedy Spanner algorithm.
* `research_ml.py`: Balanced Random Forest classifier optimized for high-recall pruning.
* `hyperparameter_benchmark.py`: Comparative metaheuristic solver (GA, PSO, DE, RS).
* `final_pruned_graph.html`: Interactive GIS proof-of-concept for real-world application.

---

## ⚡ Quick Start
```bash
# 1. Generate Ground Truth & Features
python data_generator.py

# 2. Train Optimized Model
python research_ml.py

# 3. Run Evolutionary Benchmark
python hyperparameter_benchmark.py