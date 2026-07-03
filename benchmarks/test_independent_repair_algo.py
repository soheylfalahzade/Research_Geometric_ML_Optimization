import pandas as pd
import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

# ۱. لود داده‌ها و ساخت گراف اصلی
df = pd.read_csv("final_scientific_spanner.csv")

G_orig = nx.DiGraph()
for _, row in df.iterrows():
    G_orig.add_edge(int(row['node_u']), int(row['node_v']), length=row['length'])

# شبیه‌سازی لبه‌های هرس شده توسط GNN (به عنوان مثال ۳۰٪ لبه‌ها هرس تصادفی شده‌اند)
np.random.seed(42)
all_edges = [{"u": int(row['node_u']), "v": int(row['node_v']), "length": row['length'], "idx": idx} for idx, row in df.iterrows()]
prune_rate = 0.30
num_pruned = int(len(all_edges) * prune_rate)

pruned_indices = np.random.choice(len(all_edges), num_pruned, replace=False)
E_pruned = [all_edges[i] for i in pruned_indices]
E_keep = [all_edges[i] for i in range(len(all_edges)) if i not in pruned_indices]

# ساخت گراف هرس شده اولیه
G_sparse = nx.DiGraph()
G_sparse.add_nodes_from(G_orig.nodes())
for edge in E_keep:
    G_sparse.add_edge(edge["u"], edge["v"], length=edge["length"])

def _check_one_directed_edge(args):
    repaired_G, edge, t_limit = args
    u, v, w = edge["u"], edge["v"], edge["length"]
    try:
        dist = nx.shortest_path_length(repaired_G, u, v, weight="length")
        return edge["idx"], dist > t_limit * w
    except:
        return edge["idx"], True

# الگوریتم معمولی فعلی شما (با مشکل بیش‌بازسازی)
def glv_repair_naive(G_sparse, removed_edges, t_limit=1.5, batch_size=250):
    repaired_G = G_sparse.copy()
    repairs = []
    candidates = sorted(removed_edges, key=lambda e: e["length"])
    
    for start_idx in range(0, len(candidates), batch_size):
        batch = candidates[start_idx : start_idx + batch_size]
        args_list = [(repaired_G, edge, t_limit) for edge in batch]
        
        with ThreadPoolExecutor(max_workers=4) as pool:
            needs_repair_map = dict(pool.map(_check_one_directed_edge, args_list))
            
        for edge in batch:
            if needs_repair_map.get(edge["idx"], False):
                repaired_G.add_edge(edge["u"], edge["v"], length=edge["length"])
                repairs.append(edge["idx"])
    return repaired_G, repairs

# الگوریتم جدید پیشنهادی (Spatially Independent Batch Repair)
def glv_repair_independent(G_sparse, removed_edges, t_limit=1.5, batch_size=250):
    repaired_G = G_sparse.copy()
    repairs = []
    candidates = sorted(removed_edges, key=lambda e: e["length"])
    
    while candidates:
        batch = []
        used_nodes = set()
        remaining_candidates = []
        
        # گلچین کردن لبه‌هایی که هیچ گره مشترکی ندارند
        for edge in candidates:
            u, v = edge["u"], edge["v"]
            if len(batch) < batch_size and u not in used_nodes and v not in used_nodes:
                batch.append(edge)
                used_nodes.add(u)
                used_nodes.add(v)
            else:
                remaining_candidates.append(edge)
                
        if not batch:
            break
            
        args_list = [(repaired_G, edge, t_limit) for edge in batch]
        with ThreadPoolExecutor(max_workers=4) as pool:
            needs_repair_map = dict(pool.map(_check_one_directed_edge, args_list))
            
        for edge in batch:
            if needs_repair_map.get(edge["idx"], False):
                repaired_G.add_edge(edge["u"], edge["v"], length=edge["length"])
                repairs.append(edge["idx"])
                
        candidates = remaining_candidates
        
    return repaired_G, repairs

# اجرای هر دو روش
print("Running Naive Repair (Current Buggy)...")
_, repairs_naive = glv_repair_naive(G_sparse, E_pruned)

print("Running Spatially Independent Repair (Proposed)...")
_, repairs_indep = glv_repair_independent(G_sparse, E_pruned)

print("\n================== COMPARISON RESULTS ==================")
print(f"Total Edges Initially Pruned: {len(E_pruned)}")
print(f"Naive Repair - Restored Edges: {len(repairs_naive)} (Sparsification Kept: {((len(E_pruned)-len(repairs_naive))/len(df))*100:.2f}%)")
print(f"Independent Repair - Restored Edges: {len(repairs_indep)} (Sparsification Kept: {((len(E_pruned)-len(repairs_indep))/len(df))*100:.2f}%)")
print(f"SPARSIFICATION IMPROVEMENT: {(((len(repairs_naive) - len(repairs_indep)) / len(df)) * 100):.2f}% absolute more edges pruned!")
print("========================================================")
