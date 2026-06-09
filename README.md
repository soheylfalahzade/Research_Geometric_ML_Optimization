# Geospatial Graph Optimizer: AI-Driven t-Spanner Construction

An AI-driven optimization framework that utilizes Machine Learning and Metaheuristic Evolutionary Algorithms to prune and construct high-quality geometric $t$-spanner graphs on real-world urban topologies.

---

## 📊 Performance & Optimization Results
### 1. Evolutionary Benchmark (GA vs. PSO vs. DE vs. RS)
We benchmarked four metaheuristic strategies to optimize hyperparameters. The Genetic Algorithm (GA) achieved the superior Fitness Score, demonstrating faster convergence and higher stability.
![Benchmark Convergence](benchmark_convergence.png)

### 2. Topological Pruning Visualization (Eindhoven, NL)
Interactive GIS visualization showing the AI-pruned network. The model successfully identified and pruned redundant edges while maintaining the integrity of the critical routing arteries.
![Pruned Graph](final_pruned_graph.png)

### 3. Confusion Matrix & Feature Importance
Our Random Forest classifier, tuned by GA, demonstrates high precision in identifying "Spanner Edges" vs. "Pruned Edges".
![Confusion Matrix](final_research_matrix.png)

---

## 🚀 Key Performance Indicators (KPIs)
* **Best Fitness (GA):** 0.7868 (ROC-AUC: 0.8427, Recall: 0.7159)
* **Pruning Efficiency:** ~30% reduction in edge density.
* **Query Acceleration:** 14x latency reduction for shortest-path queries.

---

## 🛠️ System Architecture
1. **Exact Ground Truth:** Generates exact $t$-spanners using Greedy Spanner ($t=2.0$).
2. **Metaheuristic Optimization:** Employs GA to fine-tune model hyperparameters for maximum pruning recall.
3. **Global Topological Features:** Incorporates **PageRank** and **Edge Centrality** to capture graph-wide significance.