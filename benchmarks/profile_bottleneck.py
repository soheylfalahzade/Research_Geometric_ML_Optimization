"""
profile_bottleneck.py
======================
هدف: پیدا کردن اینکه دقیقاً کدوم بخش از پایپ‌لاین GNN بیشترین زمان رو
می‌گیره — خود شبکه‌ی عصبی، محاسبه‌ی PageRank، یا مرحله‌ی GLV-Repair.

این اطلاعات تعیین می‌کنه که آیا باید:
  (الف) پیاده‌سازی رو بهینه کنیم (اگه مشکل از overhead غیرضروریه)، یا
  (ب) روایت مقاله رو عوض کنیم (اگه مشکل ذاتی الگوریتمه)

نحوه‌ی اجرا:
    python benchmarks/profile_bottleneck.py --city Rome
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import networkx as nx
import osmnx as ox
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q1_glv_ultimate_optimizer import (  # noqa: E402
    GeometricEdgeSAGE, glv_repair_directed, CITIES, PRUNING_THRESHOLD, MC_SAMPLES, GLV_T_LIMIT
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = PROJECT_ROOT / "results" / "models" / "best_base_model.pt"


def timed(label):
    """context manager ساده برای اندازه‌گیری و چاپ زمان هر بخش"""
    class _Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.t0
            print(f"  ⏱  {label}: {elapsed:.4f}s")
            _Timer.last = elapsed
    return _Timer()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="Rome")
    args = parser.parse_args()

    city_query = CITIES[args.city]
    print(f"Profiling city: {args.city}\n")

    with timed("1. Download graph (OSMnx)"):
        G_raw = ox.graph_from_place(city_query, network_type="drive")

    with timed("2. SCC extraction"):
        nodes_raw = list(G_raw.nodes())
        adj_raw = nx.adjacency_matrix(G_raw, nodelist=nodes_raw, weight="length")
        n_components, labels = connected_components(adj_raw, directed=True, connection="strong")
        largest_cc_idx = np.argmax(np.bincount(labels))
        nodes_to_keep = [nodes_raw[i] for i in range(len(nodes_raw)) if labels[i] == largest_cc_idx]
        G = G_raw.subgraph(nodes_to_keep).copy()
    print(f"  → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    with timed("3. PageRank computation"):
        pagerank = nx.pagerank(G, weight="length")

    with timed("4. Feature engineering (StandardScaler, tensors)"):
        edge_list = list(G.edges(data=True))
        node_map = {n: i for i, n in enumerate(G.nodes())}
        raw_node = np.array([[G.in_degree(n), G.out_degree(n), pagerank.get(n, 1e-4)] for n in G.nodes()])
        x_local = torch.tensor(StandardScaler().fit_transform(raw_node), dtype=torch.float)
        edge_idx = torch.tensor(
            [[node_map[u] for u, v, d in edge_list], [node_map[v] for u, v, d in edge_list]],
            dtype=torch.long,
        )
        raw_ea = np.array([[d["length"]] for u, v, d in edge_list])
        edge_attr_loc = torch.tensor(StandardScaler().fit_transform(raw_ea), dtype=torch.float)

    model = GeometricEdgeSAGE()
    model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True, map_location="cpu"))

    with timed(f"5. MC-Dropout inference ({MC_SAMPLES} forward passes)"):
        model.train()
        mc_preds = []
        with torch.no_grad():
            for _ in range(MC_SAMPLES):
                p = torch.sigmoid(model(x_local, edge_idx, edge_attr_loc)).numpy().flatten()
                mc_preds.append(p)
        mean_probs = np.mean(mc_preds, axis=0)
        std_probs = np.std(mc_preds, axis=0)
        calibrated_probs = mean_probs + (0.5 * std_probs)

    with timed("6. Build sparse graph + identify removal candidates"):
        G_sparse = nx.DiGraph()
        G_sparse.add_nodes_from(G.nodes())
        removed = []
        for i, (u, v, d) in enumerate(edge_list):
            is_bottleneck = (G.in_degree(v) <= 1) or (G.out_degree(u) <= 1)
            if calibrated_probs[i] > PRUNING_THRESHOLD or is_bottleneck:
                G_sparse.add_edge(u, v, length=d["length"])
            else:
                removed.append({"u": u, "v": v, "length": d["length"], "idx": i})
    print(f"  → {len(removed)} candidate edges for removal")

    with timed("7. GLV-Repair (the directed batch-sequential repair loop)"):
        G_final, repairs = glv_repair_directed(G_sparse, removed, t_limit=GLV_T_LIMIT)
    print(f"  → {len(repairs)} edges restored")

    print("\n" + "=" * 60)
    print("خلاصه: اگه مرحله‌ی ۷ (GLV-Repair) بیشترین زمان رو گرفته باشه،")
    print("یعنی گلوگاه واقعی، تأیید هندسی است، نه خود شبکه‌ی عصبی.")
    print("این دقیقاً چیزیه که باید در مقاله (یا در بهینه‌سازی بعدی) لحاظ بشه.")
    print("=" * 60)


if __name__ == "__main__":
    main()
    