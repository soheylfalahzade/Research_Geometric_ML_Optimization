import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time

# ۱. لود داده‌ها
df = pd.read_csv("final_scientific_spanner.csv")
G_orig = nx.DiGraph()
for _, row in df.iterrows():
    G_orig.add_edge(int(row['node_u']), int(row['node_v']), length=row['length'])

all_edges = [{"u": int(row['node_u']), "v": int(row['node_v']), "length": row['length'], "idx": idx} for idx, row in df.iterrows()]

def _check_one_directed_edge(args):
    repaired_G, edge, t_limit = args
    u, v, w = edge["u"], edge["v"], edge["length"]
    try:
        dist = nx.shortest_path_length(repaired_G, u, v, weight="length")
        return edge["idx"], dist > t_limit * w
    except:
        return edge["idx"], True

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

def glv_repair_independent(G_sparse, removed_edges, t_limit=1.5, batch_size=250):
    repaired_G = G_sparse.copy()
    repairs = []
    candidates = sorted(removed_edges, key=lambda e: e["length"])
    while candidates:
        batch = []
        used_nodes = set()
        remaining_candidates = []
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

# جاروب پارامتریک روی حد مجاز ضریب کشش برای باز کردن فضا برای هرس
t_limits = [1.5, 2.0, 2.5, 3.0]
naive_kept = []
indep_keep = []
routing_speedups = []

print("Starting parametric sweep to generate Q1 trade-off plots...")

# شبیه‌سازی هرس ۵۰ درصدی اولیه لبه‌ها توسط GNN
np.random.seed(42)
prune_rate = 0.50
num_pruned = int(len(all_edges) * prune_rate)
pruned_indices = np.random.choice(len(all_edges), num_pruned, replace=False)
E_pruned = [all_edges[i] for i in pruned_indices]
E_keep = [all_edges[i] for i in range(len(all_edges)) if i not in pruned_indices]

G_sparse = nx.DiGraph()
G_sparse.add_nodes_from(G_orig.nodes())
for edge in E_keep:
    G_sparse.add_edge(edge["u"], edge["v"], length=edge["length"])

for t in t_limits:
    print(f"Evaluating t_limit = {t}...")
    
    # اجرای ترمیم حریصانه موازی معمولی
    G_naive, repairs_naive = glv_repair_naive(G_sparse, E_pruned, t_limit=t)
    naive_kept.append(((len(E_pruned) - len(repairs_naive)) / len(df)) * 100)
    
    # اجرای ترمیم مستقل مکانی جدید
    G_indep, repairs_indep = glv_repair_independent(G_sparse, E_pruned, t_limit=t)
    indep_keep.append(((len(E_pruned) - len(repairs_indep)) / len(df)) * 100)
    
    # تست سرعت مسیریابی نهایی روی گراف جهت‌دار واقعی (۱۰۰۰ کوئری تصادفی)
    common_nodes = list(G_orig.nodes())
    t_orig_total = 0
    t_span_total = 0
    
    # نمونه‌برداری ۱۰۰۰ تایی برای اثبات ادعای شتاب مسیریابی نهایی
    for _ in range(1000):
        u, v = np.random.choice(common_nodes, 2, replace=False)
        try:
            t0 = time.time()
            _ = nx.shortest_path_length(G_orig, u, v, weight='length')
            t_orig_total += (time.time() - t0)
            
            t0 = time.time()
            _ = nx.shortest_path_length(G_indep, u, v, weight='length')
            t_span_total += (time.time() - t0)
        except:
            continue
            
    speedup = (t_orig_total / t_span_total) if t_span_total > 0 else 1.0
    routing_speedups.append(speedup)

# --- رسم نمودار ۱: منحنی مقایسه درصد هرس نهایی دو روش ---
plt.figure(figsize=(10, 6))
plt.plot(t_limits, naive_kept, marker='o', linestyle='--', color='crimson', label='Naive Parallel Repair (Current)')
plt.plot(t_limits, indep_keep, marker='s', linestyle='-', color='navy', label='Spatially Independent Repair (Proposed)')
plt.title("Sparsification vs. Stretch Limit Trade-off Curve", fontsize=12, fontweight='bold')
plt.xlabel("Multiplicative Stretch Factor Limit (t)")
plt.ylabel("Final Sparsification Kept (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("q1_tradeoff_curve.png", dpi=300)
print("✓ Saved q1_tradeoff_curve.png")

# --- رسم نمودار ۲: شتاب واقعی مسیریابی در کاربرد نهایی (Downstream speedup) ---
plt.figure(figsize=(10, 6))
plt.plot(t_limits, routing_speedups, marker='^', linestyle='-', color='forestgreen', label='Dijkstra Routing Speedup')
plt.title("Final Downstream Routing Acceleration in ITS Applications", fontsize=12, fontweight='bold')
  
plt.xlabel("Multiplicative Stretch Factor Limit (t)")
plt.ylabel("Search Speedup Factor (x Times Faster)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("q1_routing_speedup.png", dpi=300)
print("✓ Saved q1_routing_speedup.png")

print("\nAll plots generated successfully. Ready to verify.")
