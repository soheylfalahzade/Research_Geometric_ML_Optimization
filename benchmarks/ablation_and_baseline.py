"""
ablation_and_baseline.py (نسخه‌ی ۲ — با Edge Betweenness Centrality واقعی)
=============================================================================
همان ۴ حالت قبلی (Full Proposed / No MC-Dropout / Non-GA Threshold /
Random Pruning Baseline)، این‌بار با ویژگی centrality واقعی در مدل.

نحوه‌ی اجرا:
    python benchmarks/ablation_and_baseline.py --city Eindhoven --seeds 5
    python benchmarks/ablation_and_baseline.py --city Rome --seeds 3
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx
import osmnx as ox
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import dijkstra, connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q1_glv_ultimate_optimizer import (  # noqa: E402
    GeometricEdgeSAGE, glv_repair_directed, compute_edge_centrality, build_edge_features,
    CITIES, MC_SAMPLES, GLV_T_LIMIT
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_RUNS_DIR = PROJECT_ROOT / "results" / "raw_runs"
RAW_RUNS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_PATH = PROJECT_ROOT / "results" / "models" / "best_base_model.pt"

OUT_CSV = RAW_RUNS_DIR / f"ablation_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_city_scc(city_query):
    G_raw = ox.graph_from_place(city_query, network_type="drive")
    nodes_raw = list(G_raw.nodes())
    adj_raw = nx.adjacency_matrix(G_raw, nodelist=nodes_raw, weight="length")
    _, labels = connected_components(adj_raw, directed=True, connection="strong")
    largest_cc_idx = np.argmax(np.bincount(labels))
    nodes_to_keep = [nodes_raw[i] for i in range(len(nodes_raw)) if labels[i] == largest_cc_idx]
    return G_raw.subgraph(nodes_to_keep).copy()


def prepare_features(G):
    """ویژگی‌های گره (degree+pagerank) و یال (length+centrality واقعی)."""
    log("  Computing PageRank...")
    pagerank = nx.pagerank(G, weight="length")
    edge_list = list(G.edges(data=True))
    node_map = {n: i for i, n in enumerate(G.nodes())}
    raw_node = np.array([[G.in_degree(n), G.out_degree(n), pagerank.get(n, 1e-4)] for n in G.nodes()])
    x_local = torch.tensor(StandardScaler().fit_transform(raw_node), dtype=torch.float)
    edge_idx = torch.tensor(
        [[node_map[u] for u, v, d in edge_list], [node_map[v] for u, v, d in edge_list]],
        dtype=torch.long,
    )
    log("  Computing Edge Betweenness Centrality (this can take a bit for large cities)...")
    centrality_dict = compute_edge_centrality(G, weight="length")
    edge_attr_loc = build_edge_features(edge_list, centrality_dict)
    return edge_list, x_local, edge_idx, edge_attr_loc


def compute_stretch_and_scc(G_orig, G_final, seed=42, num_sources=500):
    nodes = list(G_orig.nodes())
    matrix_orig = nx.adjacency_matrix(G_orig, nodelist=nodes, weight="length")
    matrix_final = nx.adjacency_matrix(G_final, nodelist=nodes, weight="length")
    rng = np.random.default_rng(seed)
    sources = rng.choice(len(nodes), size=min(len(nodes), num_sources), replace=False)
    dist_orig = dijkstra(matrix_orig, directed=True, indices=sources)
    dist_final = dijkstra(matrix_final, directed=True, indices=sources)
    valid = (dist_orig > 0) & (dist_orig < np.inf) & (dist_final < np.inf)
    stretches = dist_final[valid] / dist_orig[valid]
    adj_final = nx.adjacency_matrix(G_final, weight="length")
    n_scc, _ = connected_components(adj_final, directed=True, connection="strong")
    return stretches, n_scc


def build_and_repair(G, edge_list, keep_mask, t_limit=GLV_T_LIMIT):
    G_sparse = nx.DiGraph()
    G_sparse.add_nodes_from(G.nodes())
    removed = []
    for i, (u, v, d) in enumerate(edge_list):
        is_bottleneck = (G.in_degree(v) <= 1) or (G.out_degree(u) <= 1)
        if keep_mask[i] or is_bottleneck:
            G_sparse.add_edge(u, v, length=d["length"])
        else:
            removed.append({"u": u, "v": v, "length": d["length"], "idx": i})
    G_final, repairs = glv_repair_directed(G_sparse, removed, t_limit=t_limit)
    return G_final, repairs


def variant_full_proposed(model, x_local, edge_idx, edge_attr_loc, threshold, seed):
    torch.manual_seed(seed)
    model.train()
    mc_preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            mc_preds.append(torch.sigmoid(model(x_local, edge_idx, edge_attr_loc)).numpy().flatten())
    mean_probs = np.mean(mc_preds, axis=0)
    std_probs = np.std(mc_preds, axis=0)
    calibrated = mean_probs + 0.5 * std_probs
    return calibrated > threshold


def variant_no_mc_dropout(model, x_local, edge_idx, edge_attr_loc, threshold, seed):
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_local, edge_idx, edge_attr_loc)).numpy().flatten()
    return probs > threshold


def variant_random_threshold(model, x_local, edge_idx, edge_attr_loc, seed, bad_threshold=0.50):
    return variant_full_proposed(model, x_local, edge_idx, edge_attr_loc, bad_threshold, seed)


def variant_random_pruning_baseline(edge_list, target_keep_fraction, seed):
    rng = np.random.default_rng(seed)
    n = len(edge_list)
    return rng.random(n) < target_keep_fraction


def run_all_variants(city_label, city_query, model, threshold, n_seeds):
    log(f"\n{'='*60}\n{city_label}\n{'='*60}")
    G = load_city_scc(city_query)
    edge_list, x_local, edge_idx, edge_attr_loc = prepare_features(G)
    n_edges = len(edge_list)
    log(f"{G.number_of_nodes()} nodes, {n_edges} edges")

    rows = []
    for seed in range(n_seeds):
        mask = variant_full_proposed(model, x_local, edge_idx, edge_attr_loc, threshold, seed)
        G_final, repairs = build_and_repair(G, edge_list, mask)
        stretches, scc = compute_stretch_and_scc(G, G_final, seed=seed)
        keep_fraction = mask.mean()
        rows.append({
            "city": city_label, "variant": "Full_Proposed (MC-Dropout + GA-threshold + Centrality)", "seed": seed,
            "sparsification_pct": (1 - keep_fraction) * 100,
            "max_stretch": float(np.max(stretches)), "n_repairs": len(repairs),
            "scc_connected": scc == 1,
        })

        mask_nomc = variant_no_mc_dropout(model, x_local, edge_idx, edge_attr_loc, threshold, seed)
        G_final2, repairs2 = build_and_repair(G, edge_list, mask_nomc)
        stretches2, scc2 = compute_stretch_and_scc(G, G_final2, seed=seed)
        rows.append({
            "city": city_label, "variant": "Ablation_No_MC_Dropout", "seed": seed,
            "sparsification_pct": (1 - mask_nomc.mean()) * 100,
            "max_stretch": float(np.max(stretches2)), "n_repairs": len(repairs2),
            "scc_connected": scc2 == 1,
        })

        mask_rt = variant_random_threshold(model, x_local, edge_idx, edge_attr_loc, seed)
        G_final3, repairs3 = build_and_repair(G, edge_list, mask_rt)
        stretches3, scc3 = compute_stretch_and_scc(G, G_final3, seed=seed)
        rows.append({
            "city": city_label, "variant": "Ablation_Non_GA_Threshold(0.50)", "seed": seed,
            "sparsification_pct": (1 - mask_rt.mean()) * 100,
            "max_stretch": float(np.max(stretches3)), "n_repairs": len(repairs3),
            "scc_connected": scc3 == 1,
        })

        mask_rand = variant_random_pruning_baseline(edge_list, keep_fraction, seed)
        G_final4, repairs4 = build_and_repair(G, edge_list, mask_rand)
        stretches4, scc4 = compute_stretch_and_scc(G, G_final4, seed=seed)
        rows.append({
            "city": city_label, "variant": "Baseline_Random_Pruning(matched_sparsity)", "seed": seed,
            "sparsification_pct": (1 - mask_rand.mean()) * 100,
            "max_stretch": float(np.max(stretches4)), "n_repairs": len(repairs4),
            "scc_connected": scc4 == 1,
        })

        log(f"  seed={seed}: Full={rows[-4]['max_stretch']:.3f} (repairs={rows[-4]['n_repairs']}) | "
            f"NoMC={rows[-3]['max_stretch']:.3f} (repairs={rows[-3]['n_repairs']}) | "
            f"RandThresh={rows[-2]['max_stretch']:.3f} (repairs={rows[-2]['n_repairs']}) | "
            f"RandPrune={rows[-1]['max_stretch']:.3f} (repairs={rows[-1]['n_repairs']})")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="Eindhoven")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.40)
    args = parser.parse_args()

    model = GeometricEdgeSAGE()
    model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True, map_location="cpu"))

    rows = run_all_variants(args.city, CITIES[args.city], model, args.threshold, args.seeds)

    df = pd.DataFrame(rows)
    write_header = not OUT_CSV.exists()
    df.to_csv(OUT_CSV, mode="a", header=write_header, index=False)

    print("\n" + "=" * 70)
    print("خلاصه‌ی میانگین هر Variant:")
    print(df.groupby("variant")[["sparsification_pct", "max_stretch", "n_repairs"]].mean().to_string())
    print("=" * 70)
    print(f"\nذخیره شد در: {OUT_CSV}")


if __name__ == "__main__":
    main()