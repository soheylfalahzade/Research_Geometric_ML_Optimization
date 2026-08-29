"""
تشخیص سریع: آیا گراف اصلی (MultiDiGraph از OSMnx) یال‌های موازی دارد،
و آیا nx.adjacency_matrix وزن این یال‌های موازی را جمع می‌کند (به‌جای
گرفتن کمینه)؟ این می‌تواند علت ریشه‌ای stretch < 1.0 در
monte_carlo_validator.py باشد.
"""
import networkx as nx
import osmnx as ox
from scipy.sparse.csgraph import connected_components
import numpy as np

CITY = "Eindhoven, Netherlands"

print(f"[Diag] Loading {CITY} (drive network)...")
G_raw = ox.graph_from_place(CITY, network_type="drive")
print(f"[Diag] Graph type: {type(G_raw).__name__}")

nodes_raw = list(G_raw.nodes())
adj_raw = nx.adjacency_matrix(G_raw, nodelist=nodes_raw, weight='length')
n_components, labels = connected_components(adj_raw, directed=True, connection='strong')
largest_cc_idx = np.argmax(np.bincount(labels))
nodes_to_keep = [nodes_raw[i] for i in range(len(nodes_raw)) if labels[i] == largest_cc_idx]
G = G_raw.subgraph(nodes_to_keep).copy()
print(f"[Diag] After SCC extraction: {type(G).__name__}, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

from collections import Counter
pair_counts = Counter()
pair_lengths = {}
for u, v, k, d in G.edges(keys=True, data=True):
    pair_counts[(u, v)] += 1
    pair_lengths.setdefault((u, v), []).append(d.get('length', None))

num_parallel_pairs = sum(1 for c in pair_counts.values() if c > 1)
print(f"[Diag] Total distinct (u,v) directed pairs: {len(pair_counts)}")
print(f"[Diag] Pairs with >1 parallel edge: {num_parallel_pairs}")

if num_parallel_pairs > 0:
    example_pair = next(p for p, c in pair_counts.items() if c > 1)
    lengths = pair_lengths[example_pair]
    print(f"[Diag] Example pair {example_pair}: parallel edge lengths = {lengths}")
    print(f"[Diag]   -> true shortest single-edge length (min) = {min(lengths):.2f}")
    print(f"[Diag]   -> sum of all parallel lengths          = {sum(lengths):.2f}")

    u_ex, v_ex = example_pair
    node_to_idx = {n: i for i, n in enumerate(nodes_to_keep)}
    matrix_orig = nx.adjacency_matrix(G, nodelist=nodes_to_keep, weight='length')
    val_in_matrix = matrix_orig[node_to_idx[u_ex], node_to_idx[v_ex]]
    print(f"[Diag]   -> value nx.adjacency_matrix() actually stored = {val_in_matrix:.2f}")

    if abs(val_in_matrix - sum(lengths)) < 1e-6:
        print("[Diag] CONFIRMED BUG: adjacency_matrix SUMS parallel edge weights,")
        print("        not the minimum. This artificially inflates 'original' shortest")
        print("        path distances wherever a parallel edge exists, which can make")
        print("        the spanner (simple DiGraph, no parallel edges) appear SHORTER")
        print("        than the 'original' graph on the same hop -> stretch < 1.0.")
    elif abs(val_in_matrix - min(lengths)) < 1e-6:
        print("[Diag] adjacency_matrix took the MINIMUM (not summed) - hypothesis A not confirmed for this case.")
    else:
        print(f"[Diag] adjacency_matrix value doesn't match sum or min exactly - needs closer look (got {val_in_matrix:.2f}, sum={sum(lengths):.2f}, min={min(lengths):.2f}).")
else:
    print("[Diag] No parallel edges found in this city after SCC extraction - hypothesis A ruled out for Eindhoven, need another explanation.")
