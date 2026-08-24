import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))

import pyrosm
import networkx as nx
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components, dijkstra
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

from spanner_pipeline import (
    GeometricEdgeSAGE, train_base_model, compute_edge_centrality,
    build_edge_features, glv_repair_directed, DEFAULT_MODEL_PT,
    PRUNING_THRESHOLD, MC_SAMPLES
)

CENTRALITY_K_SCALE_TEST = 10

def load_tokyo_graph(pbf_path='/home/soheil79/osm_data/Tokyo.osm.pbf'):
    print("[Scale Test] Loading Tokyo from local pbf...")
    osm = pyrosm.OSM(pbf_path)
    nodes, edges = osm.get_network(network_type='driving', nodes=True)
    G_raw = osm.to_graph(nodes, edges, graph_type='networkx')
    G_raw = nx.MultiDiGraph(G_raw)
    return G_raw

def main():
    t_start = time.time()
    WEIGHTS = DEFAULT_MODEL_PT
    model = GeometricEdgeSAGE()
    model.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location="cpu"))

    G_raw = load_tokyo_graph()
    nodes_raw = list(G_raw.nodes())
    print(f"[Scale Test] Raw graph: {len(nodes_raw)} nodes, {G_raw.number_of_edges()} edges")

    t0 = time.time()
    adj_raw = nx.adjacency_matrix(G_raw, nodelist=nodes_raw, weight='length')
    n_components, labels = connected_components(adj_raw, directed=True, connection='strong')
    largest_cc_idx = np.argmax(np.bincount(labels))
    nodes_to_keep = [nodes_raw[i] for i in range(len(nodes_raw)) if labels[i] == largest_cc_idx]
    G = G_raw.subgraph(nodes_to_keep).copy()
    print(f"[Scale Test] SCC extraction: {len(nodes_raw)} -> {G.number_of_nodes()} nodes ({time.time()-t0:.1f}s)")

    pagerank = nx.pagerank(G, weight='length')
    edge_list = list(G.edges(data=True))
    node_map = {n: i for i, n in enumerate(G.nodes())}
    raw_node = np.array([[G.in_degree(n), G.out_degree(n), pagerank.get(n, 1e-4)] for n in G.nodes()])
    x_local = torch.tensor(StandardScaler().fit_transform(raw_node), dtype=torch.float)
    edge_idx = torch.tensor([[node_map[u] for u, v, d in edge_list],
                             [node_map[v] for u, v, d in edge_list]], dtype=torch.long)

    t0 = time.time()
    centrality_dict = compute_edge_centrality(G, weight="length", k=CENTRALITY_K_SCALE_TEST)
    print(f"[Scale Test] Centrality (k={CENTRALITY_K_SCALE_TEST}) done in {time.time()-t0:.1f}s")

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
    for i, (u, v, d) in enumerate(edge_list):
        is_bottleneck = (G.in_degree(v) <= 1) or (G.out_degree(u) <= 1)
        if calibrated_probs[i] > PRUNING_THRESHOLD or is_bottleneck:
            G_sparse.add_edge(u, v, length=d["length"])
        else:
            removed_edges.append({"u": u, "v": v, "length": d["length"], "prob": mean_probs[i], "idx": i})

    print(f"[Scale Test] Running Directed GLV-Repair ({len(removed_edges)} candidates)...")
    t0 = time.time()
    G_final, repaired_idx = glv_repair_directed(G_sparse, removed_edges)
    print(f"[Scale Test] GLV-Repair done in {time.time()-t0:.1f}s")

    print(f"[Scale Test] Full pipeline (pre-verification) total time: {time.time()-t_start:.1f}s")
    print(f"[Scale Test] Full graph edges: {G.number_of_edges()}, Spanner edges: {G_final.number_of_edges()}, "
          f"Pruned: {(1 - G_final.number_of_edges()/G.number_of_edges())*100:.2f}%")

    # Independent verification (directed, sampled Monte Carlo, same as monte_carlo_validator.py)
    t0 = time.time()
    nodes = list(G.nodes())
    matrix_orig = nx.adjacency_matrix(G, nodelist=nodes, weight='length')
    matrix_final = nx.adjacency_matrix(G_final, nodelist=nodes, weight='length')
    rng = np.random.default_rng(42)
    num_sources = min(len(nodes), 500)
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
    print(f"[Scale Test] Verification done in {time.time()-t0:.1f}s on {len(stretches)} pairs")
    print(f"[Scale Test] Mean: {stretches.mean():.4f}, Median: {np.median(stretches):.4f}, "
          f"Max: {stretches.max():.4f}, Violations(>1.5): {(stretches > 1.5).mean()*100:.4f}%")


if __name__ == '__main__':
    main()
