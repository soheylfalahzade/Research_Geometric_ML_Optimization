# Geospatial Graph Optimizer: AI-Driven t-Spanner Construction for ITS

An AI-driven optimization framework that utilizes Machine Learning and Metaheuristic Evolutionary Algorithms to predict, prune, and construct high-quality geometric $t$-spanner graphs for real-world urban topologies.

---

## 📊 Performance Benchmarks & Methodology

### 1. Convergence Comparison
The Genetic Algorithm (GA) outperformed Particle Swarm (PSO) and Differential Evolution (DE) in finding optimal pruning hyperparameters.
![Benchmark Convergence](benchmark_convergence.png)

### 2. Topological Performance Summary
| Algorithm | Best Fitness | ROC-AUC | Pruned Edge Recall |
| :--- | :--- | :--- | :--- |
| **Genetic Algorithm (GA)** | **0.7868** | 0.8427 | **0.7159** |
| Particle Swarm (PSO) | 0.7836 | 0.8350 | 0.7010 |
| Differential Evolution (DE) | 0.7814 | 0.8310 | 0.6980 |
| Random Search (RS) | 0.7791 | 0.8250 | 0.6850 |

---

## 📐 Complexity & Theoretical Analysis

### Algorithmic Bottleneck
The construction of a $t$-spanner graph via the greedy approach entails a complexity of:
$$O(m \cdot (n + m) \log n)$$
where $n$ is nodes and $m$ is edges. Our proposed ML-based pruning reduces the edge count $m$ to $m'$ where $m' \ll m$, accelerating queries to:
$$O(m' + n \log n)$$

### ML-Based Optimization
We define a classifier $f_\theta$ mapping topological features to the spanner set:
$$\hat{y}_{i} = f_\theta(length_i, centrality_i, PageRank_{u,v})$$
This approach effectively prunes ~30% of the network edges while preserving 88% of topological connectivity.

---

## 🚀 Future Integration: Dynamic Rerouting
Our optimized graph serves as the backbone for the [Adaptive ITS Framework](https://github.com/soheylfalahzade/adaptive-its-priority), enabling real-time emergency vehicle preemption in CARLA/SUMO by allowing 14x faster path recalculations.

---

## 🛠️ Repository Structure
* `data_generator.py`: Generates spatial dataset and computes exact greedy ground truth.
* `research_ml.py`: Balanced Random Forest classifier.
* `genetic_optimizer.py`: Evolutionary search for optimal hyperparameters.
* `final_pruned_graph.html`: Interactive GIS visualization of the optimized road topology.