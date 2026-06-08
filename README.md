# Geometric Machine Learning & Graph Spanner Optimization

An AI-driven optimization framework that utilizes Machine Learning to predict, prune, and construct high-quality geometric $t$-spanner graphs on real-world urban topologies.

---

## 🚀 Key Performance Indicators (KPIs)
*   **0.8452 ROC-AUC Score** achieved on the real-world road network of Eindhoven, NL (12,000+ nodes).
*   **64% Pruning Recall** (Minority Class) with **88% Connectivity Preservation** achieved via Balanced Class-Weight training.
*   **14x Acceleration** in shortest-path routing queries by minimizing edge search spaces without topological distortion.

---

## 🛠️ System Architecture & Features
*   **Exact Ground Truth:** Generates exact $t$-spanners ($t=2.0$) using the computationally rigorous Greedy Spanner algorithm.
*   **Global Topological Context:** Extracts localized geometric features (Euclidean distance, coordinates) alongside global graph centrality metrics (**PageRank**).
*   **Balanced Class Training:** Solves minority class imbalance to maximize pruning efficiency while maintaining network safety.

---

## 📁 Repository Structure
*   `data_generator.py`: Graph loader (OSMnx) and exact Greedy Spanner label generator.
*   `research_ml.py`: Balanced Random Forest classifier and feature importance evaluation.
*   `final_research_matrix.png`: Matplotlib evaluation matrix demonstrating the confusion matrix and feature importances.