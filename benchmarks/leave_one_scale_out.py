import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pyrosm
import networkx as nx
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.preprocessing import StandardScaler

from spanner_pipeline import (
    GeometricEdgeSAGE, train_base_model, city_worker,
    finetune_on_repairs_continual, compute_edge_centrality,
    build_edge_features, glv_repair_directed, CITIES,
    DEFAULT_MODEL_PT, DEFAULT_FIGURES_DIR, PRUNING_THRESHOLD, MC_SAMPLES,
    safe_weighted_adjacency
)

TOKYO_PBF = '/home/soheil79/osm_data/Tokyo.osm.pbf'
MEXICO_PBF = '/home/soheil79/osm_data/MexicoCity.osm.pbf'
CENTRALITY_K_LARGE_SCALE = 40


def load_local_pbf_as_result(pbf_path, city_label, model):
    print(f"\n[{city_label}] Loading from local pbf: {pbf_path}")
    osm = pyrosm.OSM(pbf_path)
    nodes, edges = osm.get_network(network_type='driving', nodes=True)
    G_raw = nx.MultiDiGraph(osm.to_graph(nodes, edges, graph_type='networkx'))
    nodes_raw = list(G_raw.nodes())

    adj_raw = nx.adjacency_matrix(G_raw, nodelist=nodes_raw, weight='length')
    n_components, labels = connected_components(adj_raw, directed=True, connection='strong')
    largest_cc_idx = np.argmax(np.bincount(labels))
    nodes_to_keep = [nodes_raw[i] for i in range(len(nodes_raw)) if labels[i] == largest_cc_idx]
    G = G_raw.subgraph(nodes_to_keep).copy()
    print(f"[{city_label}] SCC: {len(nodes_raw)} -> {G.number_of_nodes()} nodes")

    pagerank = nx.pagerank(G, weight='length')
    edge_list = list(G.edges(data=True))
    node_map = {n: i for i, n in enumerate(G.nodes())}
    raw_node = np.array([[G.in_degree(n), G.out_degree(n), pagerank.get(n, 1e-4)] for n in G.nodes()])
    x_local = torch.tensor(StandardScaler().fit_transform(raw_node), dtype=torch.float)
    edge_idx = torch.tensor([[node_map[u] for u, v, d in edge_list],
                             [node_map[v] for u, v, d in edge_list]], dtype=torch.long)

    t0 = time.time()
    centrality_dict = compute_edge_centrality(G, weight="length", k=CENTRALITY_K_LARGE_SCALE)
    print(f"[{city_label}] Centrality (k={CENTRALITY_K_LARGE_SCALE}) done in {time.time()-t0:.1f}s")
    edge_attr_loc = build_edge_features(edge_list, centrality_dict)

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
    removed_edges = []
    MIN_SAFE_DEGREE = 2
    live_out_degree = dict(G.out_degree())
    live_in_degree = dict(G.in_degree())
    for i, (u, v, d) in enumerate(edge_list):
        is_bottleneck = (live_in_degree[v] <= MIN_SAFE_DEGREE) or (live_out_degree[u] <= MIN_SAFE_DEGREE)
        if calibrated_probs[i] > PRUNING_THRESHOLD or is_bottleneck:
            G_sparse.add_edge(u, v, length=d["length"])
        else:
            removed_edges.append({"u": u, "v": v, "length": d["length"], "prob": mean_probs[i], "idx": i})
            live_out_degree[u] -= 1
            live_in_degree[v] -= 1

    print(f"[{city_label}] Running GLV-Repair ({len(removed_edges)} candidates)...")
    G_repaired, repaired_indices = glv_repair_directed(G_sparse, removed_edges)

    return {
        "city_label": city_label, "repaired_idx": repaired_indices, "probs": mean_probs.tolist(),
        "_G": G, "_G_repaired": G_repaired, "_edge_list": edge_list,
        "_x_local": x_local, "_edge_idx": edge_idx, "_edge_attr_loc": edge_attr_loc
    }


def evaluate_city_with_model(model, res, num_samples=100000):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            model(res["_x_local"], res["_edge_idx"], res["_edge_attr_loc"])
        ).numpy().flatten()

    G = res["_G"]
    edge_list = res["_edge_list"]
    G_opt = nx.DiGraph()
    G_opt.add_nodes_from(G.nodes())
    removed = []
    for i, (u, v, d) in enumerate(edge_list):
        if probs[i] > 0.55:
            G_opt.add_edge(u, v, length=d["length"])
        else:
            removed.append({'u': u, 'v': v, 'length': d['length'], 'prob': probs[i], 'idx': i})

    G_final, _ = glv_repair_directed(G_opt, removed)

    pruned_pct = (1 - G_final.number_of_edges() / G.number_of_edges()) * 100
    print(f"[{res['city_label']}] Pruned: {pruned_pct:.2f}% ({G.number_of_edges()} -> {G_final.number_of_edges()})")

    nodes = list(G.nodes())
    matrix_orig = safe_weighted_adjacency(G, nodelist=nodes, weight='length')
    matrix_final = nx.adjacency_matrix(G_final, nodelist=nodes, weight='length')
    rng = np.random.default_rng(42)
    num_sources = min(len(nodes), 1000)
    sources = rng.choice(len(nodes), size=num_sources, replace=False)
    d_orig = dijkstra(matrix_orig, directed=True, indices=sources)
    d_span = dijkstra(matrix_final, directed=True, indices=sources)

    stretches = []
    for i in range(num_sources):
        row_o, row_s = d_orig[i], d_span[i]
        valid = (row_o > 0) & (row_o < np.inf) & (row_s < np.inf)
        if np.any(valid):
            stretches.extend((row_s[valid] / row_o[valid]).tolist())
    stretches = np.array(stretches)
    if len(stretches) > num_samples:
        idx = rng.choice(len(stretches), size=num_samples, replace=False)
        stretches = stretches[idx]

    return stretches, pruned_pct


def main():
    WEIGHTS = DEFAULT_MODEL_PT
    print("[Scale-Out] Training base model on synthetic data...")
    train_base_model(weights_path=WEIGHTS)

    model = GeometricEdgeSAGE()
    model.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location="cpu"))

    print("\n[Scale-Out] Loading 4 small cities for fine-tuning...")
    small_city_results = []
    for lbl, qry in CITIES.items():
        small_city_results.append(city_worker((lbl, qry, WEIGHTS)))
        print(f"[Scale-Out] \u2713 {lbl} loaded.")

    print("\n[Scale-Out] Loading Tokyo (large-scale, will be included in fine-tuning)...")
    tokyo_res = load_local_pbf_as_result(TOKYO_PBF, "Tokyo", model)

    print("\n[Scale-Out] Subsampling Tokyo edges for fine-tuning (memory safety)...")
    MAX_FT_EDGES = 50000
    n_tokyo_edges = tokyo_res["_edge_idx"].shape[1]
    if n_tokyo_edges > MAX_FT_EDGES:
        rng = np.random.default_rng(42)
        keep_idx = np.sort(rng.choice(n_tokyo_edges, size=MAX_FT_EDGES, replace=False))
        keep_idx_t = torch.tensor(keep_idx, dtype=torch.long)
        tokyo_res_ft = dict(tokyo_res)
        tokyo_res_ft["_edge_idx"] = tokyo_res["_edge_idx"][:, keep_idx_t]
        tokyo_res_ft["_edge_attr_loc"] = tokyo_res["_edge_attr_loc"][keep_idx_t]
        tokyo_res_ft["probs"] = [tokyo_res["probs"][i] for i in keep_idx]
        tokyo_res_ft["_edge_list"] = [tokyo_res["_edge_list"][i] for i in keep_idx]
        repaired_set_full = set(tokyo_res["repaired_idx"])
        old_to_new = {int(old_i): new_i for new_i, old_i in enumerate(keep_idx)}
        tokyo_res_ft["repaired_idx"] = [old_to_new[i] for i in repaired_set_full if i in old_to_new]
        print(f"[Scale-Out] Tokyo fine-tuning subsample: {n_tokyo_edges} -> {MAX_FT_EDGES} edges")
    else:
        tokyo_res_ft = tokyo_res

    print("\n[Scale-Out] Fine-tuning model on 4 small cities + Tokyo (subsampled)...")
    train_results = small_city_results + [tokyo_res_ft]
    model, loss_history = finetune_on_repairs_continual(model, train_results)

    print("\n[Scale-Out] Evaluating fine-tuned model on Tokyo (seen during training)...")
    tokyo_stretches, tokyo_pruned = evaluate_city_with_model(model, tokyo_res)

    print("\n[Scale-Out] Loading Mexico City (ZERO-SHOT, never seen by model)...")
    mexico_res_raw = load_local_pbf_as_result(MEXICO_PBF, "MexicoCity", model)
    print("\n[Scale-Out] Evaluating ZERO-SHOT on Mexico City...")
    mexico_stretches, mexico_pruned = evaluate_city_with_model(model, mexico_res_raw)

    results = []
    for label, stretches, pruned in [("Tokyo (fine-tuned)", tokyo_stretches, tokyo_pruned),
                                       ("MexicoCity (zero-shot)", mexico_stretches, mexico_pruned)]:
        results.append({
            "City": label,
            "Pruned %": f"{pruned:.2f}%",
            "Mean Stretch": f"{np.mean(stretches):.4f}",
            "Median Stretch": f"{np.median(stretches):.4f}",
            "Max Stretch": f"{np.max(stretches):.4f}",
            "Violation Rate (>1.5)": f"{np.mean(stretches > 1.5)*100:.4f}%"
        })

    print("\n" + "="*90)
    print("     LEAVE-ONE-SCALE-OUT RESULTS")
    print("="*90)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*90)

    df.to_csv(DEFAULT_FIGURES_DIR.parent / "raw_runs" / "leave_one_scale_out_results.csv", index=False)
    print("\n[Scale-Out] \u2713 Results saved to results/raw_runs/leave_one_scale_out_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (label, stretches) in zip(axes, [("Tokyo (fine-tuned)", tokyo_stretches),
                                               ("MexicoCity (zero-shot)", mexico_stretches)]):
        sorted_data = np.sort(stretches)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        ax.plot(sorted_data, yvals, linewidth=2.5, color='#9467bd')
        ax.axvline(x=1.5, color='red', linestyle='--', linewidth=1.5, label='Limit (t=1.5)')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel(r'Stretch Ratio $d_{H}/d_{G}$')
        ax.set_ylabel(r'P(Stretch $\leq$ x)')
        ax.grid(True, alpha=0.4)
        ax.legend(loc='lower right', fontsize=9)
        ax.text(0.03, 0.7, f"Mean: {np.mean(stretches):.3f}\nMax: {np.max(stretches):.3f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Leave-One-Scale-Out: Large-Scale Generalization', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(DEFAULT_FIGURES_DIR / 'leave_one_scale_out_cdf.png', dpi=300)
    print("[Scale-Out] \u2713 Plot saved to results/figures/leave_one_scale_out_cdf.png")


if __name__ == '__main__':
    main()
