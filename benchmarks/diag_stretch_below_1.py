import torch
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import dijkstra
from spanner_pipeline import (
    GeometricEdgeSAGE, train_base_model, city_worker,
    glv_repair_directed, CITIES, PRUNING_THRESHOLD, safe_weighted_adjacency
)

CITY_LABEL = "Eindhoven"
CITY_QUERY = CITIES[CITY_LABEL]
WEIGHTS = 'best_base_model.pt'

print("[Diag2] Loading model + city...")
model = train_base_model(weights_path=WEIGHTS)
res = city_worker((CITY_LABEL, CITY_QUERY, WEIGHTS))

G = res['_G']
edge_list = res['_edge_list']
x_local = res['_x_local']
edge_idx = res['_edge_idx']
edge_attr = res['_edge_attr_loc']

model.eval()
with torch.no_grad():
    final_probs = torch.sigmoid(model(x_local, edge_idx, edge_attr)).numpy().flatten()

THRESH = PRUNING_THRESHOLD
G_opt = nx.DiGraph()
G_opt.add_nodes_from(G.nodes())
removed = []
for i, (u, v, d) in enumerate(edge_list):
    if final_probs[i] > THRESH:
        G_opt.add_edge(u, v, length=d['length'])
    else:
        removed.append({'u': u, 'v': v, 'length': d['length'], 'prob': final_probs[i], 'idx': i})

G_final, repairs = glv_repair_directed(G_opt, removed)
print(f"[Diag2] G_final: {G_final.number_of_nodes()} nodes, {G_final.number_of_edges()} edges")

nodes = list(G.nodes())
matrix_orig = safe_weighted_adjacency(G, nodelist=nodes, weight='length')
matrix_final = nx.adjacency_matrix(G_final, nodelist=nodes, weight='length')

rng = np.random.default_rng(42)
num_sources = min(len(nodes), 1000)
sources = rng.choice(len(nodes), size=num_sources, replace=False)

dist_orig = dijkstra(matrix_orig, directed=True, indices=sources)
dist_final = dijkstra(matrix_final, directed=True, indices=sources)

print("\n[Diag3] Searching for stretch > 1.5 (theoretical bound violation) ...")
found_violation = 0
for i in range(num_sources):
    d_o = dist_orig[i]
    d_f = dist_final[i]
    valid = (d_o > 0) & (d_o < np.inf) & (d_f < np.inf)
    ratios = np.full_like(d_o, np.nan)
    ratios[valid] = d_f[valid] / d_o[valid]
    bad = np.where(ratios > 1.5)[0]
    for j in bad:
        if found_violation >= 5:
            break
        src_node = nodes[sources[i]]
        dst_node = nodes[j]
        print(f"\n[Diag3] === violation {found_violation+1}: {src_node} -> {dst_node} ===")
        print(f"[Diag3]   d_orig={d_o[j]:.2f}  d_final={d_f[j]:.2f}  stretch={ratios[j]:.4f}")
        for r in removed:
            if r['u'] == src_node and r['v'] == dst_node:
                print(f"[Diag3]   this exact edge was removed directly: length={r['length']:.2f}, prob={r['prob']:.4f}")
        # آیا این نود اصلاً در G_final هیچ یال ورودی/خروجی دارد؟ (احتمال ایزوله‌شدگی جزئی)
        print(f"[Diag3]   out-degree(src) in G_final: {G_final.out_degree(src_node)}, in-degree(dst) in G_final: {G_final.in_degree(dst_node)}")
        found_violation += 1
    if found_violation >= 5:
        break

if found_violation == 0:
    print("[Diag3] No violation found among these 1000 sources (may only appear in the full 100k sample).")
else:
    print(f"\n[Diag3] Total violations found (capped at 5 printed): checked {num_sources} sources.")
