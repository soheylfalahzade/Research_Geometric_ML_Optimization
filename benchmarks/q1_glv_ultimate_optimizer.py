"""
================================================================================
Q1 GEOSPATIAL GRAPH OPTIMIZER — Directed Neural-Algorithmic Spanner Pipeline
================================================================================
Target Journal: Q1 (IEEE Transactions on Intelligent Transportation Systems)
Status: Under active development; not yet submitted
Author: Soheyl Falahzade

نسخه‌ی به‌روزشده: Edge Betweenness Centrality اکنون به‌عنوان یک ویژگی
واقعی ورودی مدل استفاده می‌شود (طبق معادله‌ی ۲ در مقاله)، نه فقط برای
وزن‌دهی loss.
================================================================================
Scientific Contributions:
  1. Directed Spanner Construction (Gap 7): Handling one-way road constraints.
  2. Strongly Connected Components (SCC): Ensuring 100% urban reachability.
  3. MC-Dropout Uncertainty (Gap 9): Probabilistic edge pruning via SAGE-Dropout.
  4. Continual Learning (Gap 3): Knowledge retention via Replay Buffer.
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

# مسیرها را نسبت به محل خود فایل حساب می‌کنیم، نه نسبت به پوشه‌ای که
# اسکریپت از آنجا اجرا می‌شود — قبلاً اگر کسی از پوشه‌ای غیر از ریشهٔ
# ریپو اجرا می‌کرد، فایل CSV پیدا نمی‌شد و مدل بی‌صدا با وزن‌های خام
# (تصادفی) جایگزین می‌شد (نقض قانون ۷: قابل بازتولید بودن).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_CSV = REPO_ROOT / "results" / "data" / "spanner_dataset_pro.csv"
DEFAULT_MODEL_PT = REPO_ROOT / "results" / "models" / "best_base_model.pt"
DEFAULT_FIGURES_DIR = REPO_ROOT / "results" / "figures"
DEFAULT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# ۰. تنظیمات استراتژیک و هایپرپارامترها
# ──────────────────────────────────────────────
CITIES = {
    "Eindhoven": "Eindhoven, Netherlands",
    "Manhattan": "Manhattan, New York, USA",
    "Paris":     "Paris, France",
    "Rome":      "Rome, Italy",
}

NUM_DIJKSTRA_THREADS = 4   # تعداد تردها برای محاسبات موازی دایجسترا
GLV_T_LIMIT          = 1.5 # حد تئوریک ضریب کشش (t-Spanner)
FINETUNE_EPOCHS      = 5   # کاهش اپوک‌ها برای جلوگیری از Overwriting مدل (Gap 3)
FINETUNE_LR          = 0.0001
STRETCH_SAMPLES      = 100000 # نمونه‌برداری عظیم مونت‌کارلو
GLV_BATCH_SIZE       = 250    # اندازه دسته‌ها در ترمیم ترتیبی
PRUNING_THRESHOLD    = 0.40   # آستانه بهینه پارتو
MC_SAMPLES           = 10     # تعداد دفعات استنتاج برای MC-Dropout (Gap 9)
REPLAY_BUFFER_SIZE   = 5      # تعداد تجربه‌های ذخیره شده از هر شهر (Gap 3)
CENTRALITY_K_SAMPLES = 500    # تعداد گره‌ی نمونه برای تخمین Edge Betweenness Centrality


# ──────────────────────────────────────────────
# ۰.۵ محاسبه‌ی Edge Betweenness Centrality
# ──────────────────────────────────────────────
def compute_edge_centrality(G, weight="length", k=CENTRALITY_K_SAMPLES, seed=42):
    """
    محاسبه‌ی تخمینی Edge Betweenness Centrality با نمونه‌برداری از k گره
    مبدا (برای گراف‌های بزرگ مثل Rome، محاسبه‌ی دقیق O(V*E) غیرعملی است).
    این تابع فقط یک‌بار به‌ازای هر شهر باید صدا زده شود (نه به‌ازای هر
    seed)، چون به مدل یا seed بستگی ندارد.
    """
    n = G.number_of_nodes()
    k_actual = min(k, n) if k else None
    return nx.edge_betweenness_centrality(G, k=k_actual, weight=weight, seed=seed, normalized=True)


def build_edge_features(edge_list, centrality_dict):
    """
    می‌سازد آرایه‌ی ۲ ستونی نرمال‌شده [length, centrality] برای استفاده
    به‌عنوان edge_attr در GeometricEdgeSAGE — دقیقاً معادل بخشی از
    معادله‌ی (۲) در مقاله: [d̄(u,v), ..., C(u,v)]
    """
    raw_ea = []
    for (u, v, d) in edge_list:
        c = centrality_dict.get((u, v), 0.0)
        raw_ea.append([d["length"], c])
    raw_ea = np.array(raw_ea)
    return torch.tensor(StandardScaler().fit_transform(raw_ea), dtype=torch.float)


# ──────────────────────────────────────────────
# ۱. معماری مدل عصبی گراف آگاه از عدم قطعیت
# ──────────────────────────────────────────────
class GeometricEdgeSAGE(nn.Module):
    """
    معماری Edge-based SAGE با لایه Dropout فعال در زمان تست جهت تخمین عدم قطعیت اپیستمیک.
    ورودی گره: [In-Degree, Out-Degree, PageRank]
    ورودی یال (edge_attr_dim=2): [Normalized Length, Edge Betweenness Centrality]
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
            nn.Dropout(p=dropout_rate), # لایه حیاتی برای MC-Sampling (Gap 9)
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        # لایه‌های کانوولوشنال گراف
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)

        # ترکیب ویژگی‌های دو سر یال به همراه [طول, centrality]
        u, v = edge_index
        edge_repr = torch.cat([h[u], h[v], edge_attr], dim=-1)
        return self.classifier(edge_repr)


def q1_balanced_loss(pred_logits, target, importance, lambda_p=2.0, alpha_s=0.5):
    """
    تابع هزینه ترکیبی: BCE + جریمه ترمیم هندسی + پاداش هرس هوشمند
    """
    bce     = F.binary_cross_entropy_with_logits(pred_logits, target)
    probs   = torch.sigmoid(pred_logits)
    penalty = lambda_p * (importance * target * (1 - probs)).mean()
    sparse  = alpha_s * probs.mean()
    return bce + penalty + sparse


# ──────────────────────────────────────────────
# ۲. پروتکل آموزش پایه (Full Base Training Loop)
# ──────────────────────────────────────────────
def train_base_model(csv_path=DEFAULT_DATA_CSV, weights_path=DEFAULT_MODEL_PT):
    print("\n[Base Training] Loading spatial dataset and engineering features...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"دیتاست ضروری پیدا نشد: {csv_path}\n"
            f"این خطا عمداً fatal است (نه فقط هشدار) چون ادامهٔ کار بدون این "
            f"داده یعنی مدل با وزن‌های تصادفی/خام کار می‌کند و نتایج بی‌معنی "
            f"تولید می‌شود — طبق قانون ۷ (قابلیت بازتولید)."
        )

    df = pd.read_csv(csv_path)
    # مهندسی ویژگی برای گراف جهت‌دار
    features = ["length", "u_degree", "v_degree", "edge_centrality", "u_pagerank", "v_pagerank"]
    df[features] = StandardScaler().fit_transform(df[features])

    nodes_unique = pd.concat([df["node_u"], df["node_v"]]).unique()
    node_map     = {n: i for i, n in enumerate(nodes_unique)}
    edge_index   = torch.tensor([[node_map[u] for u in df["node_u"]],
                                 [node_map[v] for v in df["node_v"]]], dtype=torch.long)

    node_feat_map = {}
    for _, row in df.iterrows():
        # ویژگی‌های ۳تایی برای هر گره
        node_feat_map[node_map[row["node_u"]]] = [row["u_degree"], row["u_degree"], row["u_pagerank"]]
        node_feat_map[node_map[row["node_v"]]] = [row["v_degree"], row["v_degree"], row["v_pagerank"]]

    x = torch.tensor([node_feat_map.get(i, [0.0, 0.0, 0.0]) for i in range(len(nodes_unique))], dtype=torch.float)    
    y = torch.tensor(df["is_spanner_edge"].values, dtype=torch.float).view(-1, 1)

    # edge_attr حالا شامل [length, edge_centrality] واقعی است — نه فقط
    # طول. این دقیقاً همان چیزی است که مقاله در معادله‌ی (۲) ادعا می‌کند.
    # (df از قبل با StandardScaler نرمال شده، پس مستقیم قابل استفاده است.)
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
        loss = q1_balanced_loss(out, y, importance)
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
# ۳. پروتکل ترمیم جهت‌دار و دایجسترا موازی (Gap 7)
#    نسخه‌ی بهینه‌شده با cutoff واقعی (طبق بخش Systems Engineering مقاله)
# ──────────────────────────────────────────────
def _check_one_directed_edge(args):
    """بررسی نیاز به ترمیم یک یال با cutoff-bounded Dijkstra جهت‌دار."""
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
    ترمیم مستقل مکانی (Spatially Independent Batch Repair) جهت حل باگ بیش‌بازسازی موازی
    و افزایش نرخ هرس گراف با رعایت قوانین جهت‌دار.
    """
    repaired_G = G_sparse.copy()
    repairs = []
    candidates = sorted(removed_edges_info, key=lambda e: e["length"])

    while candidates:
        batch = []
        used_nodes = set()
        remaining_candidates = []

        # انتخاب لبه‌هایی که تداخل گره مکانی با هم ندارند
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
# ۳.۵ ارزیابی آماری ۱۰۰,۰۰۰ نمونه‌ای فوق‌سریع SciPy
# ──────────────────────────────────────────────
def compute_global_stretch_scipy(G_orig, G_final, num_samples=STRETCH_SAMPLES):
    """محاسبه فوق‌سریع ضریب کشش جهت‌دار با کدهای بهینه‌سازی شده C++ در SciPy"""
    nodes = list(G_orig.nodes())
    num_nodes = len(nodes)
    if num_nodes < 2: return np.array([1.0])

    matrix_orig = nx.adjacency_matrix(G_orig, nodelist=nodes, weight="length")
    matrix_final = nx.adjacency_matrix(G_final, nodelist=nodes, weight="length")

    rng = np.random.default_rng(42)
    sources = rng.choice(num_nodes, size=min(num_nodes, 500), replace=False)

    dist_orig = dijkstra(matrix_orig, directed=True, indices=sources)
    dist_span = dijkstra(matrix_final, directed=True, indices=sources)

    stretches = []
    asymmetry_data = []

    for i in range(len(sources)):
        d_o = dist_orig[i]
        d_s = dist_span[i]
        valid_mask = (d_o > 0) & (d_o < np.inf) & (d_s < np.inf)
        if not np.any(valid_mask): continue

        stretches.extend((d_s[valid_mask] / d_o[valid_mask]).tolist())
        rev_dist = dijkstra(matrix_final, directed=True, indices=[sources[i]])
        asymmetry_data.append(np.abs(d_s[valid_mask] - rev_dist[0][valid_mask]).mean())

    if len(stretches) > num_samples:
        stretches = rng.choice(stretches, size=num_samples, replace=False)

    return np.array(stretches), asymmetry_data


# ──────────────────────────────────────────────
# ۴. تابع Worker شهر — استخراج SCC، Centrality، و MC-Dropout (Gap 7 & 9)
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

    for i, (u, v, d) in enumerate(edge_list):
        is_bottleneck = (G.in_degree(v) <= 1) or (G.out_degree(u) <= 1)

        if calibrated_probs[i] > PRUNING_THRESHOLD or is_bottleneck:
            G_sparse.add_edge(u, v, length=d["length"])
        else:
            removed_edges.append({"u": u, "v": v, "length": d["length"], "prob": mean_probs[i], "idx": i})

    print(f"[{city_label}] Running Directed GLV-Repair ({len(removed_edges)} candidates)...")
    G_repaired, repaired_indices = glv_repair_directed(G_sparse, removed_edges)

    return {
        "city_label": city_label, "repaired_idx": repaired_indices, "probs": mean_probs.tolist(),
        "_G": G, "_G_repaired": G_repaired, "_edge_list": edge_list,
        "_x_local": x_local, "_edge_idx": edge_idx, "_edge_attr_loc": edge_attr_loc
    }


# ──────────────────────────────────────────────
# ۵. یادگیری مداوم با Replay Buffer (Gap 3)
# ──────────────────────────────────────────────
def finetune_on_repairs_continual(model, city_results):
    print("\n[Main] Starting Continual Learning Fine-Tuning (Gap 3 Active)...")
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
# ۶. استخراج متریک‌های نهایی و اثبات بصری جهت‌داری
# ──────────────────────────────────────────────
def compute_final_metrics_directed(model, res):
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
    for i, (u, v, d) in enumerate(edge_list):
        is_bottleneck = (G.in_degree(v) <= 1) or (G.out_degree(u) <= 1)
        if final_probs[i] > PRUNING_THRESHOLD or is_bottleneck:
            G_opt.add_edge(u, v, length=d["length"])
        else:
            removed.append({"u": u, "v": v, "length": d["length"], "prob": final_probs[i], "idx": i})

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
# ۷. تابع Main — ارکستر نهایی و رسم نمودارها
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
    print("      FINAL Q1 MASTER BENCHMARK — DIRECTED GEOMETRIC FEEDBACK LOOP")
    print("=" * 100)
    df_final = pd.DataFrame(final_results)
    df_show = df_final[['City', 'Sparsification', 'SCC', 'Avg Stretch', 'Max Stretch', 'Repairs']]
    print(df_show.to_string(index=False))
    print("=" * 100)

    print("\nGenerating final publication-quality plots...")

    plt.figure(figsize=(10, 6))
    for res in final_results:
        sorted_s = np.sort(res['_stretches'])
        y_vals = np.arange(len(sorted_s)) / float(len(sorted_s)-1)
        plt.plot(sorted_s, y_vals, label=f"{res['City']} (Max: {res['Max Stretch']})", lw=2.5)
    plt.axvline(1.5, color='red', linestyle='--', label='Limit (t=1.5)')
    plt.title("Empirical CDF of Directed Global Stretch", fontsize=14, fontweight='bold')
    plt.xlabel(r"Stretch Ratio ($d_{H}/d_{G}$)"); plt.ylabel(r"P(Stretch $\leq$ x)"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(DEFAULT_FIGURES_DIR / 'q1_global_stretch_cdf.png', dpi=300)

    plt.figure(figsize=(10, 6))
    for city, losses in loss_history.items():
        plt.plot(range(1, len(losses)+1), losses, marker='o', lw=2, label=city)
    plt.title("Continual Learning Stability via Memory Buffer", fontsize=14, fontweight='bold')
    plt.xlabel("Fine-tuning Epochs"); plt.ylabel("Total Combined Loss"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(DEFAULT_FIGURES_DIR / 'q1_continual_learning_loss.png', dpi=300)

    plt.figure(figsize=(10, 6))
    for res in final_results:
        plt.hist(res['_asymmetry'], bins=30, alpha=0.5, label=f"{res['City']} Asymmetry")
    plt.title("Visual Proof of Directed Traffic Asymmetry (Gap 7)", fontsize=14, fontweight='bold')
    plt.xlabel(r"Mean Route Difference |d(u,v) - d(v,u)| (meters)"); plt.ylabel("Frequency"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(DEFAULT_FIGURES_DIR / 'q1_directed_asymmetry_proof.png', dpi=300)

    print("\n✓ Pipeline run complete. 3 plots saved to disk.")

if __name__ == "__main__":
    main()