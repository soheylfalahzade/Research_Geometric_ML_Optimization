# Geospatial Graph Optimizer: A Hybrid Neural-Algorithmic Framework for city-scale $t$-Spanner Construction

[![Research Status: Q1 Candidate](https://img.shields.io/badge/Research-Q1--Target-gold.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)

## 📖 Abstract & Research Problem
Real-time routing in Intelligent Transportation Systems (ITS) is often hindered by the $O(V^3)$ computational complexity of constructing exact geometric $t$-spanners on massive urban topologies. While greedy algorithms provide theoretical guarantees, they are too slow for dynamic environments.

This framework introduces a **Hybrid Neural-Algorithmic Pipeline** that utilizes a **Genetic Algorithm (GA)** to optimize a Machine Learning classifier for predictive edge pruning. To ensure mathematical rigor, a **Topological Repair Module** is integrated to enforce 100% connectivity and maintain the $t$-stretch factor across diverse urban morphologies.

---

## 🚀 Key Performance Indicators (KPIs)
*   **Maximum Efficiency:** Achieved **~30% reduction** in total edge density while preserving 100.00% connectivity.
*   **Routing Acceleration:** Up to **9.5x Speedup** in shortest-path query latency (e.g., Paris topology).
*   **Theoretical Safety:** 100% adherence to the $t$-spanner property ($t=2.0$) via the Hybrid Repair Module.
*   **Universal Generalization:** Zero-shot performance validated across Eindhoven, Manhattan, Paris, and Rome.

---

## 📊 Scientific Benchmarks & Validation

### 1. Evolutionary Hyperparameter Optimization
We benchmarked four metaheuristic strategies to find the optimal pruning balance. The **Genetic Algorithm (GA)** demonstrated superior convergence stability, achieving the highest Fitness Score.

| Metaheuristic | Best Fitness Score | Status |
| :--- | :--- | :--- |
| **Genetic Algorithm (GA)** | **0.7868** | **Winner** |
| Particle Swarm (PSO) | 0.7836 | Competitor |
| Differential Evolution (DE) | 0.7814 | Competitor |
| Random Search (RS) | 0.7791 | Baseline |

![Benchmark Convergence](benchmark_convergence.png)

### 2. Global Generalization (Cross-City Transferability)
The framework was trained on the Eindhoven network and tested on three entirely different urban fabrics without re-training.

| City | Urban Fabric | Edge Reduction | Search Speedup | Avg. Stretch ($t$) |
| :--- | :--- | :--- | :--- | :--- |
| **Eindhoven** | Hybrid (Modern) | 95.1% | **4.5x** | 1.465 |
| **Manhattan** | Grid (Regular) | 64.6% | **3.5x** | 1.593 |
| **Paris** | Radial (Circular) | 98.8% | **9.5x** | 1.318 |
| **Rome** | Organic (Ancient) | 58.7% | **3.7x** | 1.527 |

### 3. Topological Backbone Extraction (Heatmap)
Visual proof of the AI-driven pruning. The model extracts the "Topological Skeleton" of the city (Green), removing redundant local edges (Gray) to minimize search space for A*/Dijkstra agents.

![Connectivity Heatmap](Q1_Comparison_Heatmap.png)

---

## 📐 Mathematical Formulation & Complexity Analysis

### I. The Computational Bottleneck
Constructing a classic Greedy $t$-Spanner requires solving the Shortest Path Problem for every edge $(u, v) \in E$:
$$Complexity_{Greedy} = O(m \cdot (n + m) \log n)$$

### II. AI-Driven Pruning Strategy
We employ a feature-based classifier $f_\theta$ mapping topological features (PageRank, Edge Centrality) to the spanner inclusion set $S$:
$$\hat{y}_{i} = f_\theta(length_i, PageRank_{u,v}, Centrality_i)$$
This shifts the heavy computation to an offline training phase, allowing for $O(m)$ online sparsification.

### III. Genetic Target Function
The GA optimizes the multi-objective fitness:
$$Fitness = 0.5 \cdot ROC\_AUC + 0.5 \cdot Recall_{Pruned}$$

---

## 🛠️ Repository Structure
*   `data_generator.py`: Generates spatial datasets from OSMnx & computes exact ground truth.
*   `genetic_optimizer.py`: Custom GA implementation for hyperparameter search.
*   `hyperparameter_benchmark.py`: Comparative suite for metaheuristic convergence.
*   `q1_final_experiment.py`: Main validation script for connectivity & repair logic.
*   `q1_global_routing_benchmark.py`: Global city-scale routing performance tester.
*   `q1_topological_cleaner.py`: Post-processing pipeline for degree-1 node removal.

---

## ⚡ Quick Start & Reproduction
1.  **Install dependencies:** `pip install osmnx networkx scikit-learn pandas matplotlib`
2.  **Generate base data:** `python data_generator.py`
3.  **Run optimization:** `python genetic_optimizer.py`
4.  **Execute global benchmark:** `python q1_global_routing_benchmark.py`

---
**Contact & Affiliation**
*   **Author:** Soheyl Falahzade
*   **Researcher:** Yazd University / Salman Farsi University of Kazerun
*   **LinkedIn:** [linkedin.com/in/soheyl-falah-zade](https://linkedin.com/in/soheyl-falah-zade)
*   **GitHub:** [github.com/soheylfalahzade](https://github.com/soheylfalahzade)
