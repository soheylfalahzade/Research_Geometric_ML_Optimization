"""
================================================================================
GEOSPATIAL GRAPH OPTIMIZER — Directed Neural-Algorithmic Spanner Pipeline
================================================================================
Status: under active development
Author: Soheil Fallahzadeh

Updated version: Edge Betweenness Centrality is now used as a real model input
feature (as the edge-feature definition requires), not only for
weighting the loss.
================================================================================
Scientific Contributions:
  1. Directed Spanner Construction: Handling one-way road constraints.
  2. Strongly Connected Components (SCC): Ensuring 100% urban reachability.
  3. MC-Dropout Uncertainty: Probabilistic edge pruning via SAGE-Dropout.
  4. Continual Learning: Knowledge retention via Replay Buffer.
  5. Batch-Sequential Repair: Solving the "Blind Batch-Pruning" Paradox.
  6. Edge Betweenness Centrality as a genuine model input (not just loss weight).
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import time
import os
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra, connected_components
import matplotlib.pyplot as plt

# Paths are resolved relative to this file's location, not to the folder
# the script is launched from — previously, if someone ran it from a folder other than the root
# of the repository, the CSV file was not found and the model was silently replaced by raw
# (random) weights (violating the reproducibility requirement).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_CSV = REPO_ROOT / "results" / "data" / "spanner_dataset_pro.csv"
DEFAULT_MODEL_PT = REPO_ROOT / "results" / "models" / "best_base_model.pt"
DEFAULT_FIGURES_DIR = REPO_ROOT / "results" / "figures"
DEFAULT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 0. Strategic settings and hyperparameters
# ──────────────────────────────────────────────
CITIES = {
    "Eindhoven": "Eindhoven, Netherlands",
    "Manhattan": "Manhattan, New York, USA",
    "Paris":     "Paris, France",
    "Rome":      "Rome, Italy",
}

NUM_DIJKSTRA_THREADS = 4   # number of threads for the parallel Dijkstra computations
GLV_T_LIMIT          = 1.5 # theoretical stretch bound (t-Spanner)
FINETUNE_EPOCHS      = 5   # fewer epochs to avoid overwriting the model
FINETUNE_LR          = 0.0001
STRETCH_SAMPLES      = 100000 # large-scale Monte Carlo sampling
GLV_BATCH_SIZE       = 250    # batch size of the sequential repair
PRUNING_THRESHOLD    = 0.40   # Pareto-optimal threshold
MC_SAMPLES           = 10     # number of inference passes for MC-Dropout
REPLAY_BUFFER_SIZE   = 5      # number of stored experiences per city
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

CENTRALITY_K_SAMPLES = 80    # number of sampled nodes for estimating Edge Betweenness Centrality


# ──────────────────────────────────────────────
# 0.5 Computation of Edge Betweenness Centrality
# ──────────────────────────────────────────────
def compute_edge_centrality(G, weight="length", k=CENTRALITY_K_SAMPLES, seed=42):
    """
    Approximate computation of Edge Betweenness Centrality by sampling k source
    nodes (for large graphs such as Rome, the exact O(V*E) computation is impractical).
    This function should be called only once per city (not per
    seed), because it does not depend on the model or the seed.
    """
    n = G.number_of_nodes()
    k_actual = min(k, n) if k else None
    return nx.edge_betweenness_centrality(G, k=k_actual, weight=weight, seed=seed, normalized=True)


def build_edge_features(edge_list, centrality_dict):
    """
    builds a normalized 2-column array [length, centrality] to be used
    as edge_attr in GeometricEdgeSAGE — exactly equivalent to part of
    the edge feature definition: [d̄(u,v), ..., C(u,v)]
    """
    raw_ea = []
    for (u, v, d) in edge_list:
        c = centrality_dict.get((u, v), 0.0)
        raw_ea.append([d["length"], c])
    raw_ea = np.array(raw_ea)
    return torch.tensor(StandardScaler().fit_transform(raw_ea), dtype=torch.float)


# ──────────────────────────────────────────────
# 1. Architecture of the uncertainty-aware graph neural network
# ──────────────────────────────────────────────
class GeometricEdgeSAGE(nn.Module):
    """
    Edge-based SAGE architecture with Dropout kept active at test time to estimate epistemic uncertainty.
    Node input: [In-Degree, Out-Degree, PageRank]
    Edge input (edge_attr_dim=2): [Normalized Length, Edge Betweenness Centrality]
    """
    def __init__(self, in_channels=3, hidden_channels=64, dropout_rate=0.2, edge_attr_dim=2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.edge_attr_dim = edge_attr_dim
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + edge_attr_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), # critical layer for MC-Sampling
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        # Graph convolutional layers
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)

        # Combine the features of the two edge endpoints with [length, centrality]
        u, v = edge_index
        edge_repr = torch.cat([h[u], h[v], edge_attr], dim=-1)
        return self.classifier(edge_repr)


def balanced_loss(pred_logits, target, importance, lambda_p=2.0, alpha_s=0.5):
    """
    Combined loss: BCE + geometric repair penalty + smart-pruning reward
    """
    bce     = F.binary_cross_entropy_with_logits(pred_logits, target)
    probs   = torch.sigmoid(pred_logits)
    penalty = lambda_p * (importance * target * (1 - probs)).mean()
    sparse  = alpha_s * probs.mean()
    return bce + penalty + sparse


# ──────────────────────────────────────────────
# 2. Base training protocol (Full Base Training Loop)
# ──────────────────────────────────────────────
def train_base_model(csv_path=DEFAULT_DATA_CSV, weights_path=DEFAULT_MODEL_PT):
    print("\n[Base Training] Loading spatial dataset and engineering features...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Required dataset not found: {csv_path}\n"
            f"This error is deliberately fatal (not just a warning) because continuing without this "
            f"data would run the model with random/raw weights and produce meaningless "
            f"results (the run must be reproducible)."
        )

    df = pd.read_csv(csv_path)
    # Feature engineering for the directed graph
    features = ["length", "u_degree", "v_degree", "edge_centrality", "u_pagerank", "v_pagerank"]
    df[features] = StandardScaler().fit_transform(df[features])

    nodes_unique = pd.concat([df["node_u"], df["node_v"]]).unique()
    node_map     = {n: i for i, n in enumerate(nodes_unique)}
    edge_index   = torch.tensor([[node_map[u] for u in df["node_u"]],
                                 [node_map[v] for v in df["node_v"]]], dtype=torch.long)

    node_feat_map = {}
    for _, row in df.iterrows():
        # 3-element features for each node
        node_feat_map[node_map[row["node_u"]]] = [row["u_degree"], row["v_degree"], row["u_pagerank"]]
        node_feat_map[node_map[row["node_v"]]] = [row["v_degree"], row["u_degree"], row["v_pagerank"]]

    x = torch.tensor([node_feat_map.get(i, [0.0, 0.0, 0.0]) for i in range(len(nodes_unique))], dtype=torch.float)    
    y = torch.tensor(df["is_spanner_edge"].values, dtype=torch.float).view(-1, 1)

    # edge_attr now contains the real [length, edge_centrality] — not only
    # the length. This is exactly what the edge-feature definition requires.
    # (df already normalized with StandardScaler, so it can be used directly.)
    edge_attr = torch.tensor(df[["length", "edge_centrality"]].values, dtype=torch.float)

    importance = torch.tensor(df["edge_centrality"].values, dtype=torch.float).view(-1, 1)

    model = GeometricEdgeSAGE().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    print("Starting base GNN training with Early Stopping...")
    best_loss, patience, counter = float("inf"), 20, 0

    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index, edge_attr)
        loss = balanced_loss(out, y, importance)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
            torch.save(model.state_dict(), weights_path)
        else:
            counter += 1

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:03d} | Loss: {loss.item():.4f}")
        if counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(weights_path, weights_only=True))
    print(f"Base training complete. Weights saved → {weights_path}\n")
    return model


# ──────────────────────────────────────────────
# 3. Directed repair protocol and parallel Dijkstra
#    optimized version with a true cutoff (cutoff-bounded search)
# ──────────────────────────────────────────────
def _check_one_directed_edge(args):
    """Check whether one edge needs repair using a directed cutoff-bounded Dijkstra."""
    repaired_G, edge, t_limit = args
    u, v, w = edge["u"], edge["v"], edge["length"]
    cutoff = t_limit * w
    try:
        lengths = nx.single_source_dijkstra_path_length(repaired_G, u, cutoff=cutoff, weight="length")
        return edge["idx"], v not in lengths
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return edge["idx"], True
    except Exception:
        return edge["idx"], True

def glv_repair_directed(G_sparse, removed_edges_info, t_limit=GLV_T_LIMIT):
    """
    Spatially Independent Batch Repair, to fix the parallel over-restoration bug
    and to increase the pruning rate while respecting directed edge semantics.
    """
    repaired_G = G_sparse.copy()
    repairs = []
    candidates = sorted(removed_edges_info, key=lambda e: e["length"])

    while candidates:
        batch = []
        used_nodes = set()
        remaining_candidates = []

        # Select edges that do not share spatial nodes with each other
        for edge in candidates:
            u, v = edge["u"], edge["v"]
            if len(batch) < GLV_BATCH_SIZE and u not in used_nodes and v not in used_nodes:
                batch.append(edge)
                used_nodes.add(u)
                used_nodes.add(v)
            else:
                remaining_candidates.append(edge)

        if not batch:
            break

        args_list = [(repaired_G, edge, t_limit) for edge in batch]
        with ThreadPoolExecutor(max_workers=NUM_DIJKSTRA_THREADS) as pool:
            needs_repair_map = dict(pool.map(_check_one_directed_edge, args_list))

        for edge in batch:
            if needs_repair_map.get(edge["idx"], False):
                repaired_G.add_edge(edge["u"], edge["v"], length=edge["length"])
                repairs.append(edge["idx"])

        candidates = remaining_candidates

    return repaired_G, repairs


# ──────────────────────────────────────────────
# 3.49 Fix for parallel edges (Parallel-Edge Summation Bug)
# ──────────────────────────────────────────────
def safe_weighted_adjacency(G, nodelist, weight="length"):
    """
    Builds the weighted adjacency matrix correctly even when G is a MultiDiGraph with parallel
    edges. nx.adjacency_matrix on a MultiDiGraph sums the weights of parallel
    edges instead of taking the minimum; this function keeps the minimum of each pair of parallel
    edges.
    """
    if G.is_multigraph():
        G_simple = nx.DiGraph() if G.is_directed() else nx.Graph()
        G_simple.add_nodes_from(G.nodes())
        for u, v, d in G.edges(data=True):
            w = d.get(weight, 1)
            if not G_simple.has_edge(u, v) or w < G_simple[u][v].get(weight, float("inf")):
                G_simple.add_edge(u, v, **{weight: w})
        G = G_simple
    return nx.adjacency_matrix(G, nodelist=nodelist, weight=weight)


# ──────────────────────────────────────────────
# 3.5 Ultra-fast 100,000-sample statistical evaluation in SciPy
# ──────────────────────────────────────────────
def compute_global_stretch_scipy(G_orig, G_final, num_samples=STRETCH_SAMPLES):
    """Ultra-fast computation of the directed stretch with optimized C++ code in SciPy"""
    nodes = list(G_orig.nodes())
    num_nodes = len(nodes)
    if num_nodes < 2: return np.array([1.0])

    matrix_orig = safe_weighted_adjacency(G_orig, nodelist=nodes, weight="length")
    matrix_final = nx.adjacency_matrix(G_final, nodelist=nodes, weight="length")

    rng = np.random.default_rng(42)
    sources = rng.choice(num_nodes, size=min(num_nodes, 500), replace=False)

    dist_orig = dijkstra(matrix_orig, directed=True, indices=sources)
    dist_span = dijkstra(matrix_final, directed=True, indices=sources)

    # To measure the true directed asymmetry d(u,v) versus d(v,u), the
    # distances must be computed on the transpose of the graph -- previously this line
    # ran again on the same directed graph from the same source, which returned exactly the same
    # forward distance (a difference of about zero, always), not the true return distance.
    # This made the "asymmetry evidence" plot always fall near zero
    # regardless of how asymmetric the graph really is.
    matrix_final_T = matrix_final.transpose().tocsr()
    dist_span_reverse = dijkstra(matrix_final_T, directed=True, indices=sources)

    stretches = []
    asymmetry_data = []

    for i in range(len(sources)):
        d_o = dist_orig[i]
        d_s = dist_span[i]
        valid_mask = (d_o > 0) & (d_o < np.inf) & (d_s < np.inf)
        if not np.any(valid_mask): continue

        stretches.extend((d_s[valid_mask] / d_o[valid_mask]).tolist())
        rev_dist = dist_span_reverse[i]
        both_valid = valid_mask & (rev_dist < np.inf)
        if np.any(both_valid):
            asymmetry_data.append(np.abs(d_s[both_valid] - rev_dist[both_valid]).mean())

    if len(stretches) > num_samples:
        stretches = rng.choice(stretches, size=num_samples, replace=False)

    return np.array(stretches), asymmetry_data


# ──────────────────────────────────────────────
# 4. City worker function — SCC extraction, centrality, and MC-Dropout
# ──────────────────────────────────────────────
def city_worker(args):
    city_label, city_query, weights_path = args
    print(f"\n[{city_label}] Starting Directed Processing (PID={os.getpid()})...")

    model = GeometricEdgeSAGE()
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location="cpu"))

    G_raw = ox.graph_from_place(city_query, network_type="drive")

    nodes_raw = list(G_raw.nodes())
    adj_raw = nx.adjacency_matrix(G_raw, nodelist=nodes_raw, weight='length')
    n_components, labels = connected_components(adj_raw, directed=True, connection='strong')
    largest_cc_idx = np.argmax(np.bincount(labels))
    nodes_to_keep = [nodes_raw[i] for i in range(len(nodes_raw)) if labels[i] == largest_cc_idx]
    G = G_raw.subgraph(nodes_to_keep).copy()

    print(f"[{city_label}] SCC Extraction: {len(nodes_raw)} -> {G.number_of_nodes()} nodes.")

    pagerank = nx.pagerank(G, weight='length')
    edge_list = list(G.edges(data=True))
    node_map = {n: i for i, n in enumerate(G.nodes())}

    raw_node = np.array([[G.in_degree(n), G.out_degree(n), pagerank.get(n, 1e-4)] for n in G.nodes()])
    x_local = torch.tensor(StandardScaler().fit_transform(raw_node), dtype=torch.float)
    edge_idx = torch.tensor([[node_map[u] for u, v, d in edge_list],
                             [node_map[v] for u, v, d in edge_list]], dtype=torch.long)

    print(f"[{city_label}] Computing edge betweenness centrality (k={CENTRALITY_K_SAMPLES} samples)...")
    centrality_dict = compute_edge_centrality(G, weight="length", k=CENTRALITY_K_SAMPLES)
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

    # ── live safeguard against bottlenecks (replaces the earlier static safeguard) ──
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

    print(f"[{city_label}] Running Directed GLV-Repair ({len(removed_edges)} candidates)...")
    G_repaired, repaired_indices = glv_repair_directed(G_sparse, removed_edges)

    return {
        "city_label": city_label, "repaired_idx": repaired_indices, "probs": mean_probs.tolist(),
        "_G": G, "_G_repaired": G_repaired, "_edge_list": edge_list,
        "_x_local": x_local, "_edge_idx": edge_idx, "_edge_attr_loc": edge_attr_loc
    }


# ──────────────────────────────────────────────
# 5. Continual learning with Replay Buffer
# ──────────────────────────────────────────────
def finetune_on_repairs_continual(model, city_results):
    print("\n[Main] Starting Continual Learning Fine-Tuning...")
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)

    replay_buffer = []
    loss_tracking = {}

    for res in city_results:
        label = res["city_label"]
        repaired_set = set(res["repaired_idx"])
        n_edges = len(res["_edge_list"])

        y_vals = []
        for i in range(n_edges):
            if i in repaired_set or res["probs"][i] > PRUNING_THRESHOLD:
                y_vals.append(1.0)
            else:
                y_vals.append(0.0)
        y_feedback = torch.tensor(y_vals, dtype=torch.float).view(-1, 1)

        print(f"  [{label}] Fine-tuning 5 epochs. Buffer size: {len(replay_buffer)} cities.")
        model.train()

        city_epoch_losses = []
        for ep in range(FINETUNE_EPOCHS):
            optimizer.zero_grad()

            out_current = model(res["_x_local"], res["_edge_idx"], res["_edge_attr_loc"])
            loss_current = F.binary_cross_entropy_with_logits(out_current, y_feedback)

            loss_memory = 0.0
            if replay_buffer:
                mem = random.choice(replay_buffer)
                out_mem = model(mem['x'], mem['e_idx'], mem['e_attr'])
                loss_memory = F.binary_cross_entropy_with_logits(out_mem, mem['y'])

            total_loss = loss_current + (0.5 * loss_memory)
            total_loss.backward()
            optimizer.step()
            city_epoch_losses.append(total_loss.item())

        loss_tracking[label] = city_epoch_losses

        replay_buffer.append({
            'x': res["_x_local"], 'e_idx': res["_edge_idx"],
            'e_attr': res["_edge_attr_loc"], 'y': y_feedback
        })

    print("[Main] Continual Learning session complete.\n")
    return model, loss_tracking


# ──────────────────────────────────────────────
# 6. Final metric extraction and visual evidence of directedness
# ──────────────────────────────────────────────
def compute_final_metrics_directed(model, res, disable_safeguard=False):
    label = res["city_label"]
    G, edge_list = res["_G"], res["_edge_list"]

    print(f"[{label}] Running final directed optimized inference...")
    model.train()

    with torch.no_grad():
        mc_preds = [torch.sigmoid(model(res["_x_local"], res["_edge_idx"], res["_edge_attr_loc"])).numpy().flatten() for _ in range(MC_SAMPLES)]

    final_probs = np.mean(mc_preds, axis=0) + (0.5 * np.std(mc_preds, axis=0))

    G_opt = nx.DiGraph()
    G_opt.add_nodes_from(G.nodes())
    removed = []
    MIN_SAFE_DEGREE = 2
    live_out_degree = dict(G.out_degree())
    live_in_degree = dict(G.in_degree())
    for i, (u, v, d) in enumerate(edge_list):
        is_bottleneck = False if disable_safeguard else (
            (live_in_degree[v] <= MIN_SAFE_DEGREE) or (live_out_degree[u] <= MIN_SAFE_DEGREE)
        )
        if final_probs[i] > PRUNING_THRESHOLD or is_bottleneck:
            G_opt.add_edge(u, v, length=d["length"])
        else:
            removed.append({"u": u, "v": v, "length": d["length"], "prob": final_probs[i], "idx": i})
            if not disable_safeguard:
                live_out_degree[u] -= 1
                live_in_degree[v] -= 1

    G_final, repairs = glv_repair_directed(G_opt, removed)

    stretches, asymmetry = compute_global_stretch_scipy(G, G_final)

    adj_final = nx.adjacency_matrix(G_final, weight='length')
    n_scc, _ = connected_components(adj_final, directed=True, connection='strong')

    return {
        "City": label,
        "Sparsification": f"{(1 - G_final.number_of_edges()/G.number_of_edges())*100:.2f}%",
        "SCC": "100%" if n_scc == 1 else f"{n_scc} components",
        "Avg Stretch": f"{np.mean(stretches):.3f}",
        "Max Stretch": f"{np.max(stretches):.3f}",
        "Repairs": len(repairs),
        "_stretches": stretches,
        "_asymmetry": asymmetry
    }


# ──────────────────────────────────────────────
# 7. Main function — final orchestrator and plotting
# ──────────────────────────────────────────────
def main():
    mp.set_start_method("spawn", force=True)

    WEIGHTS = DEFAULT_MODEL_PT
    model = train_base_model(weights_path=WEIGHTS)

    print("\n" + "=" * 65)
    print("PHASE 1 — Parallel Directed City Processing (Enforcing SCC)")
    print("=" * 65)
    worker_args = [(lbl, qry, WEIGHTS) for lbl, qry in CITIES.items()]
    city_results = []
    with ProcessPoolExecutor(max_workers=len(CITIES)) as pool:
        futures = {pool.submit(city_worker, arg): arg[0] for arg in worker_args}
        for f in as_completed(futures):
            city_results.append(f.result())
            print(f"[Main] ✓ {futures[f]} processing finished.")

    print("\n" + "=" * 65)
    print("PHASE 2 — Continual Learning & Experience Replay")
    print("=" * 65)
    model, loss_history = finetune_on_repairs_continual(model, city_results)

    print("\n" + "=" * 65)
    print("PHASE 3 — Final Benchmarking & Visual Proof Generation")
    print("=" * 65)
    final_results = [compute_final_metrics_directed(model, res) for res in city_results]

    print("\n" + "=" * 100)
    print("      FINAL MASTER BENCHMARK — DIRECTED GEOMETRIC FEEDBACK LOOP")
    print("=" * 100)
    df_final = pd.DataFrame(final_results)
    df_show = df_final[['City', 'Sparsification', 'SCC', 'Avg Stretch', 'Max Stretch', 'Repairs']]
    print(df_show.to_string(index=False))
    print("=" * 100)

    print("\nGenerating final plots...")

    plt.figure(figsize=(10, 6))
    for res in final_results:
        sorted_s = np.sort(res['_stretches'])
        y_vals = np.arange(len(sorted_s)) / float(len(sorted_s)-1)
        plt.plot(sorted_s, y_vals, label=f"{res['City']} (Max: {res['Max Stretch']})", lw=2.5)
    plt.axvline(1.5, color='red', linestyle='--', label='Limit (t=1.5)')
    plt.title("Empirical CDF of Directed Global Stretch", fontsize=14, fontweight='bold')
    plt.xlabel(r"Stretch Ratio ($d_{H}/d_{G}$)"); plt.ylabel(r"P(Stretch $\leq$ x)"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(DEFAULT_FIGURES_DIR / 'global_stretch_cdf.png', dpi=300)

    plt.figure(figsize=(10, 6))
    for city, losses in loss_history.items():
        plt.plot(range(1, len(losses)+1), losses, marker='o', lw=2, label=city)
    plt.title("Continual Learning Stability via Memory Buffer", fontsize=14, fontweight='bold')
    plt.xlabel("Fine-tuning Epochs"); plt.ylabel("Loss = current-city loss + 0.5 × replay-buffer loss"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(DEFAULT_FIGURES_DIR / 'continual_learning_loss.png', dpi=300)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False, sharey=False)
    colors = {'Manhattan': 'tab:blue', 'Eindhoven': 'tab:orange', 'Paris': 'tab:green', 'Rome': 'tab:red'}
    for ax, res in zip(axes.flat, final_results):
        city = res['City']
        ax.hist(res['_asymmetry'], bins=30, color=colors.get(city, 'gray'), alpha=0.85)
        ax.set_title(city, fontsize=12, fontweight='bold')
        ax.set_xlabel(r"|d(u,v) - d(v,u)| (m)")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
    fig.suptitle("Empirical Evidence of Directed Traffic Asymmetry", fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(DEFAULT_FIGURES_DIR / 'directed_asymmetry_evidence.png', dpi=300)

    print("\n✓ Pipeline run complete. 3 plots saved to disk.")

if __name__ == "__main__":
    main()