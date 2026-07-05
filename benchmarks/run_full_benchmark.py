"""
run_full_benchmark.py (نسخه‌ی ۲ — با Edge Betweenness Centrality واقعی)
==========================================================================
همان منطق نسخه‌ی قبلی، با یک تفاوت کلیدی: ویژگی centrality اکنون واقعاً
محاسبه و به مدل داده می‌شود (نه فقط طول یال). centrality فقط یک‌بار به
ازای هر شهر محاسبه می‌شود (نه به‌ازای هر seed) چون به مدل بستگی ندارد.

نحوه‌ی اجرا (از ریشه‌ی پروژه):
    python benchmarks/run_full_benchmark.py
    python benchmarks/run_full_benchmark.py --city Eindhoven --seeds 2
"""

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx
import osmnx as ox
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import dijkstra, connected_components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_RUNS_DIR = RESULTS_DIR / "raw_runs"
DATA_DIR = RESULTS_DIR / "data"
MODELS_DIR = RESULTS_DIR / "models"
RAW_RUNS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q1_glv_ultimate_optimizer import (  # noqa: E402
    GeometricEdgeSAGE,
    glv_repair_directed,
    train_base_model,
    compute_edge_centrality,
    build_edge_features,
    CITIES,
    PRUNING_THRESHOLD,
    MC_SAMPLES,
    GLV_T_LIMIT,
)

LOG_FILE = RAW_RUNS_DIR / f"benchmark_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
CSV_FILE = RAW_RUNS_DIR / f"full_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_result_row(row: dict):
    df_row = pd.DataFrame([row])
    write_header = not CSV_FILE.exists()
    df_row.to_csv(CSV_FILE, mode="a", header=write_header, index=False)


def load_city_scc(city_query: str, city_label: str) -> nx.DiGraph:
    log(f"[{city_label}] Downloading graph from OSMnx...")
    G_raw = ox.graph_from_place(city_query, network_type="drive")
    nodes_raw = list(G_raw.nodes())
    adj_raw = nx.adjacency_matrix(G_raw, nodelist=nodes_raw, weight="length")
    n_components, labels = connected_components(adj_raw, directed=True, connection="strong")
    largest_cc_idx = np.argmax(np.bincount(labels))
    nodes_to_keep = [nodes_raw[i] for i in range(len(nodes_raw)) if labels[i] == largest_cc_idx]
    G = G_raw.subgraph(nodes_to_keep).copy()
    log(f"[{city_label}] SCC extraction: {len(nodes_raw)} -> {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges.")
    return G


def run_classic_greedy_spanner(G: nx.DiGraph, t_limit: float = GLV_T_LIMIT):
    log(f"  [Greedy] Starting classic greedy construction on {G.number_of_edges()} edges...")
    t0 = time.perf_counter()

    edges = sorted(G.edges(data=True), key=lambda x: x[2]["length"])
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())

    checked = 0
    total = len(edges)
    progress_interval = max(1000, total // 20)
    for u, v, d in edges:
        w = d["length"]
        try:
            dist = nx.shortest_path_length(H, u, v, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            dist = float("inf")

        if dist > t_limit * w:
            H.add_edge(u, v, length=w)

        checked += 1
        if checked % progress_interval == 0:
            elapsed = time.perf_counter() - t0
            log(f"  [Greedy] Progress: {checked}/{total} edges "
                f"({checked/total*100:.1f}%) | elapsed: {elapsed/60:.1f} min")

    elapsed = time.perf_counter() - t0
    log(f"  [Greedy] Done in {elapsed:.2f}s ({elapsed/60:.2f} min).")
    return H, elapsed


def run_our_gnn_framework(G: nx.DiGraph, model: GeometricEdgeSAGE, edge_list, x_local, edge_idx,
                           edge_attr_loc, seed: int):
    """
    اجرای پایپ‌لاین GNN. توجه: ویژگی‌های ثابت گراف (x_local, edge_idx,
    edge_attr_loc که شامل centrality است) از بیرون پاس داده می‌شوند تا
    محاسبه‌ی سنگین centrality فقط یک‌بار به‌ازای هر شهر انجام شود، نه
    به‌ازای هر seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.perf_counter()

    model.train()
    mc_preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            p = torch.sigmoid(model(x_local, edge_idx, edge_attr_loc)).numpy().flatten()
            mc_preds.append(p)

    mean_probs = np.mean(mc_preds, axis=0)
    std_probs = np.std(mc_preds, axis=0)
    calibrated_probs = mean_probs + (0.5 * std_probs)

    G_sparse = nx.DiGraph()
    G_sparse.add_nodes_from(G.nodes())
    removed = []
    for i, (u, v, d) in enumerate(edge_list):
        is_bottleneck = (G.in_degree(v) <= 1) or (G.out_degree(u) <= 1)
        if calibrated_probs[i] > PRUNING_THRESHOLD or is_bottleneck:
            G_sparse.add_edge(u, v, length=d["length"])
        else:
            removed.append({"u": u, "v": v, "length": d["length"], "idx": i})

    G_final, repairs = glv_repair_directed(G_sparse, removed, t_limit=GLV_T_LIMIT)

    elapsed = time.perf_counter() - t0
    return G_final, elapsed, len(repairs)


def compute_stretch_and_scc(G_orig: nx.DiGraph, G_final: nx.DiGraph, num_sources: int = 500, seed: int = 42):
    nodes = list(G_orig.nodes())
    if len(nodes) < 2:
        return np.array([1.0]), 1

    matrix_orig = nx.adjacency_matrix(G_orig, nodelist=nodes, weight="length")
    matrix_final = nx.adjacency_matrix(G_final, nodelist=nodes, weight="length")

    rng = np.random.default_rng(seed)
    sources = rng.choice(len(nodes), size=min(len(nodes), num_sources), replace=False)

    dist_orig = dijkstra(matrix_orig, directed=True, indices=sources)
    dist_final = dijkstra(matrix_final, directed=True, indices=sources)

    valid = (dist_orig > 0) & (dist_orig < np.inf) & (dist_final < np.inf)
    stretches = (dist_final[valid] / dist_orig[valid])

    adj_final = nx.adjacency_matrix(G_final, weight="length")
    n_scc, _ = connected_components(adj_final, directed=True, connection="strong")

    return stretches, n_scc


def benchmark_one_city(city_label: str, city_query: str, model: GeometricEdgeSAGE, n_seeds: int):
    log(f"\n{'='*70}\nCITY: {city_label}\n{'='*70}")

    G = load_city_scc(city_query, city_label)

    # --- Greedy واقعی (یک‌بار) ---
    H_greedy, time_greedy = run_classic_greedy_spanner(G)
    spars_greedy = (1 - H_greedy.number_of_edges() / G.number_of_edges()) * 100
    stretches_greedy, scc_greedy = compute_stretch_and_scc(G, H_greedy)

    # --- ویژگی‌های ثابت گراف (شامل centrality) — فقط یک‌بار محاسبه میشه ---
    log(f"  [Features] Computing PageRank and Edge Betweenness Centrality...")
    pagerank = nx.pagerank(G, weight="length")
    edge_list = list(G.edges(data=True))
    node_map = {n: i for i, n in enumerate(G.nodes())}
    raw_node = np.array([[G.in_degree(n), G.out_degree(n), pagerank.get(n, 1e-4)] for n in G.nodes()])
    x_local = torch.tensor(StandardScaler().fit_transform(raw_node), dtype=torch.float)
    edge_idx = torch.tensor(
        [[node_map[u] for u, v, d in edge_list], [node_map[v] for u, v, d in edge_list]],
        dtype=torch.long,
    )
    centrality_dict = compute_edge_centrality(G, weight="length")
    edge_attr_loc = build_edge_features(edge_list, centrality_dict)

    # --- روش پیشنهادی (چند seed) ---
    gnn_times, gnn_spars, gnn_repairs, gnn_max_stretch, gnn_scc = [], [], [], [], []
    for seed in range(n_seeds):
        log(f"  [GNN] seed={seed} ...")
        G_final, elapsed, n_repairs = run_our_gnn_framework(
            G, model, edge_list, x_local, edge_idx, edge_attr_loc, seed
        )
        spars = (1 - G_final.number_of_edges() / G.number_of_edges()) * 100
        stretches, scc = compute_stretch_and_scc(G, G_final, seed=seed)

        gnn_times.append(elapsed)
        gnn_spars.append(spars)
        gnn_repairs.append(n_repairs)
        gnn_max_stretch.append(float(np.max(stretches)) if len(stretches) else float("nan"))
        gnn_scc.append(scc)
        log(f"  [GNN] seed={seed} | time={elapsed:.4f}s | sparsification={spars:.2f}% | "
            f"max_stretch={gnn_max_stretch[-1]:.3f} | SCC={scc}")

    mean_gnn_time = float(np.mean(gnn_times))
    std_gnn_time = float(np.std(gnn_times))
    speedup_mean = time_greedy / mean_gnn_time if mean_gnn_time > 0 else float("nan")

    row = {
        "city": city_label,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "t_limit": GLV_T_LIMIT,
        "greedy_time_sec": time_greedy,
        "greedy_sparsification_pct": spars_greedy,
        "greedy_max_stretch": float(np.max(stretches_greedy)) if len(stretches_greedy) else float("nan"),
        "greedy_scc_components": scc_greedy,
        "gnn_n_seeds": n_seeds,
        "gnn_time_mean_sec": mean_gnn_time,
        "gnn_time_std_sec": std_gnn_time,
        "gnn_sparsification_mean_pct": float(np.mean(gnn_spars)),
        "gnn_sparsification_std_pct": float(np.std(gnn_spars)),
        "gnn_max_stretch_mean": float(np.mean(gnn_max_stretch)),
        "gnn_max_stretch_worst": float(np.max(gnn_max_stretch)),
        "gnn_repairs_mean": float(np.mean(gnn_repairs)),
        "gnn_scc_all_connected": all(s == 1 for s in gnn_scc),
        "speedup_mean": speedup_mean,
        "uses_edge_centrality": True,
        "timestamp": datetime.now().isoformat(),
    }

    append_result_row(row)
    log(f"[{city_label}] ✅ Saved to {CSV_FILE.name}")
    log(f"[{city_label}] SUMMARY: Greedy={time_greedy:.2f}s | GNN={mean_gnn_time:.4f}s±{std_gnn_time:.4f} | "
        f"Speedup={speedup_mean:.2f}x")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default=None)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    log(f"Starting full benchmark run (v2, with edge centrality). Log: {LOG_FILE}")
    log(f"Results CSV: {CSV_FILE}")
    log(f"Config: t_limit={GLV_T_LIMIT}, pruning_threshold={PRUNING_THRESHOLD}, "
        f"mc_samples={MC_SAMPLES}, seeds={args.seeds}")

    weights_path = MODELS_DIR / "best_base_model.pt"
    data_csv = DATA_DIR / "spanner_dataset_pro.csv"
    if not data_csv.exists():
        alt = PROJECT_ROOT / "spanner_dataset_pro.csv"
        data_csv = alt if alt.exists() else data_csv

    # ⚠️ چون معماری مدل تغییر کرده (edge_attr_dim از ۱ به ۲)، وزن‌های
    # قدیمی دیگر سازگار نیستند. باید دوباره train شود.
    log(f"Re-training base model (architecture changed: edge_attr now includes centrality)...")
    model = train_base_model(csv_path=str(data_csv), weights_path=str(weights_path))

    cities_to_run = CITIES if args.city is None else {args.city: CITIES[args.city]}

    for city_label, city_query in cities_to_run.items():
        try:
            benchmark_one_city(city_label, city_query, model, args.seeds)
        except Exception as e:
            log(f"❌ ERROR processing {city_label}: {e}")
            log(traceback.format_exc())
            continue

    log("\n✅ ALL DONE. Final results:")
    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
        log("\n" + df.to_string(index=False))
    log(f"\nنتایج نهایی در: {CSV_FILE}")


if __name__ == "__main__":
    main()