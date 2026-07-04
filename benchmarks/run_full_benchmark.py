"""
run_full_benchmark.py
======================
هدف: جایگزینی تخمین معیوب `simulate_greedy_time_for_city` (که در
q1_master_benchmarker.py استفاده شده بود و Greedy واقعی را اجرا نمی‌کرد)
با اندازه‌گیری واقعی و صادقانه‌ی هر دو روش روی هر ۴ شهر کامل.

منطق:
  - الگوریتم Greedy Spanner واقعی (از test_sota_comparison.py، که درست
    پیاده‌سازی شده بود) روی هر شهر یک‌بار اجرا می‌شود. این الگوریتم
    قطعی (deterministic) است، پس تکرار چندباره فقط نویز اندازه‌گیری
    اضافه می‌کند، نه تنوع آماری واقعی.
  - روش پیشنهادی (GNN + MC-Dropout + GLV-Repair) به دلیل نمونه‌برداری
    تصادفی MC-Dropout، در هر seed می‌تواند کمی متفاوت باشد. پس این بخش
    را N_SEEDS بار با seedهای متفاوت اجرا می‌کنیم و mean ± std گزارش
    می‌شود.
  - نتیجه‌ی هر شهر بلافاصله پس از اتمام در CSV ذخیره می‌شود
    (checkpointing) — اگر اجرا روی Rome (که می‌تواند خیلی طول بکشد)
    قطع شد، نتایج شهرهای قبلی از دست نمی‌روند.
  - یک فایل لاگ با timestamp دقیق هر مرحله را ثبت می‌کند تا بتوانی شب
    اجرا کنی و صبح ببینی دقیقاً کجا هست یا کجا متوقف شده.

نحوه‌ی اجرا (از ریشه‌ی پروژه):
    python benchmarks/run_full_benchmark.py

    # برای اجرای فقط یک شهر خاص (مثلاً برای تست سریع قبل از Rome):
    python benchmarks/run_full_benchmark.py --city Eindhoven

    # برای تغییر تعداد seed های GNN:
    python benchmarks/run_full_benchmark.py --seeds 5

⚠️ هشدار: اجرای Greedy واقعی روی Rome (~42,000 گره) می‌تواند چند ساعت
طول بکشد. توصیه می‌شود شب اجرا شود، یا با nohup/screen در پس‌زمینه.
"""

import argparse
import json
import os
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

# ──────────────────────────────────────────────
# مسیرها — مستقل از اینکه اسکریپت از کجا اجرا میشه
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_RUNS_DIR = RESULTS_DIR / "raw_runs"
DATA_DIR = RESULTS_DIR / "data"
MODELS_DIR = RESULTS_DIR / "models"
RAW_RUNS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# اضافه کردن benchmarks/ به sys.path تا بتونیم از q1_glv_ultimate_optimizer وارد کنیم
sys.path.insert(0, str(Path(__file__).resolve().parent))

from q1_glv_ultimate_optimizer import (  # noqa: E402
    GeometricEdgeSAGE,
    glv_repair_directed,
    train_base_model,
    CITIES,
    PRUNING_THRESHOLD,
    MC_SAMPLES,
    GLV_T_LIMIT,
)

LOG_FILE = RAW_RUNS_DIR / f"benchmark_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
CSV_FILE = RAW_RUNS_DIR / f"full_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def log(msg: str):
    """چاپ در ترمینال + ذخیره در فایل لاگ با timestamp."""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_result_row(row: dict):
    """اضافه‌کردن فوری یک ردیف نتیجه به CSV (checkpointing)."""
    df_row = pd.DataFrame([row])
    write_header = not CSV_FILE.exists()
    df_row.to_csv(CSV_FILE, mode="a", header=write_header, index=False)


# ──────────────────────────────────────────────
# ۱. بارگذاری گراف شهر + استخراج بزرگ‌ترین SCC
#    (دقیقاً همون منطقی که در city_worker استفاده شده،
#    تا مقایسه‌ی Greedy و GNN روی گراف یکسان انجام بشه)
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# ۲. الگوریتم Greedy Spanner واقعی (از test_sota_comparison.py،
#    که درست پیاده‌سازی شده بود — نه شبیه‌سازی معیوب)
# ──────────────────────────────────────────────
def run_classic_greedy_spanner(G: nx.DiGraph, t_limit: float = GLV_T_LIMIT):
    log(f"  [Greedy] Starting classic greedy construction on {G.number_of_edges()} edges "
        f"(this can take a long time for large cities)...")
    t0 = time.perf_counter()

    edges = sorted(G.edges(data=True), key=lambda x: x[2]["length"])
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())

    checked = 0
    total = len(edges)
    progress_interval = max(1000, total // 20)  # حداکثر ۲۰ خط لاگ پیشرفت
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


# ──────────────────────────────────────────────
# ۳. روش پیشنهادی (GNN + MC-Dropout + GLV-Repair)
# ──────────────────────────────────────────────
def run_our_gnn_framework(G: nx.DiGraph, model: GeometricEdgeSAGE, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.perf_counter()

    pagerank = nx.pagerank(G, weight="length")
    edge_list = list(G.edges(data=True))
    node_map = {n: i for i, n in enumerate(G.nodes())}

    raw_node = np.array([[G.in_degree(n), G.out_degree(n), pagerank.get(n, 1e-4)] for n in G.nodes()])
    x_local = torch.tensor(StandardScaler().fit_transform(raw_node), dtype=torch.float)
    edge_idx = torch.tensor(
        [[node_map[u] for u, v, d in edge_list], [node_map[v] for u, v, d in edge_list]],
        dtype=torch.long,
    )
    raw_ea = np.array([[d["length"]] for u, v, d in edge_list])
    edge_attr_loc = torch.tensor(StandardScaler().fit_transform(raw_ea), dtype=torch.float)

    model.train()  # فعال نگه‌داشتن Dropout برای MC-Sampling
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


# ──────────────────────────────────────────────
# ۴. محاسبه‌ی stretch و SCC نهایی (برای اثبات درستی، نه فقط سرعت)
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# ۵. اجرای کامل برای یک شهر
# ──────────────────────────────────────────────
def benchmark_one_city(city_label: str, city_query: str, model: GeometricEdgeSAGE, n_seeds: int):
    log(f"\n{'='*70}\nCITY: {city_label}\n{'='*70}")

    G = load_city_scc(city_query, city_label)

    # --- Greedy واقعی (یک‌بار، چون قطعی است) ---
    H_greedy, time_greedy = run_classic_greedy_spanner(G)
    spars_greedy = (1 - H_greedy.number_of_edges() / G.number_of_edges()) * 100
    stretches_greedy, scc_greedy = compute_stretch_and_scc(G, H_greedy)

    # --- روش پیشنهادی (چند seed برای آمار) ---
    gnn_times, gnn_spars, gnn_repairs, gnn_max_stretch, gnn_scc = [], [], [], [], []
    for seed in range(n_seeds):
        log(f"  [GNN] seed={seed} ...")
        G_final, elapsed, n_repairs = run_our_gnn_framework(G, model, seed)
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
        "timestamp": datetime.now().isoformat(),
    }

    append_result_row(row)
    log(f"[{city_label}] ✅ Saved to {CSV_FILE.name}")
    log(f"[{city_label}] SUMMARY: Greedy={time_greedy:.2f}s | GNN={mean_gnn_time:.4f}s±{std_gnn_time:.4f} | "
        f"Speedup={speedup_mean:.2f}x")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default=None,
                         help="فقط یک شهر خاص را اجرا کن (Eindhoven/Manhattan/Paris/Rome). پیش‌فرض: همه.")
    parser.add_argument("--seeds", type=int, default=5,
                         help="تعداد seed برای اجرای روش GNN (برای mean±std).")
    args = parser.parse_args()

    log(f"Starting full benchmark run. Log: {LOG_FILE}")
    log(f"Results CSV: {CSV_FILE}")
    log(f"Config: t_limit={GLV_T_LIMIT}, pruning_threshold={PRUNING_THRESHOLD}, "
        f"mc_samples={MC_SAMPLES}, seeds={args.seeds}")

    # --- بارگذاری یا آموزش مدل پایه ---
    weights_path = MODELS_DIR / "best_base_model.pt"
    data_csv = DATA_DIR / "spanner_dataset_pro.csv"
    if not data_csv.exists():
        # fallback: شاید هنوز در ریشه‌ی پروژه باشد
        alt = PROJECT_ROOT / "spanner_dataset_pro.csv"
        data_csv = alt if alt.exists() else data_csv

    if weights_path.exists():
        log(f"Loading existing base model weights from {weights_path}")
        model = GeometricEdgeSAGE()
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location="cpu"))
    else:
        log(f"No existing weights found. Training base model from {data_csv}...")
        model = train_base_model(csv_path=str(data_csv), weights_path=str(weights_path))

    cities_to_run = CITIES if args.city is None else {args.city: CITIES[args.city]}

    for city_label, city_query in cities_to_run.items():
        try:
            benchmark_one_city(city_label, city_query, model, args.seeds)
        except Exception as e:
            log(f"❌ ERROR processing {city_label}: {e}")
            log(traceback.format_exc())
            log(f"Continuing to next city...")
            continue

    log("\n✅ ALL DONE. Final results:")
    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
        log("\n" + df.to_string(index=False))
    log(f"\nنتایج نهایی در: {CSV_FILE}")
    log("هیچ عددی رو دستی توی مقاله یا README ننویس — مستقیم از این CSV بخون.")


if __name__ == "__main__":
    main()