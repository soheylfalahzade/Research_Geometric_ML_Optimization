"""
fuzzy_pruning.py
===================
به‌جای یک آستانه‌ی سخت روی احتمال مدل (مثلاً "اگه prob > 0.40 نگه‌دار")،
یک سیستم استنتاج فازی (Mamdani-style) می‌سازیم که ۲ ورودی را ترکیب
می‌کند:
  - probability: احتمال کالیبره‌شده‌ی مدل (از MC-Dropout)
  - centrality: مرکزیت میانی یال (Edge Betweenness Centrality)

قوانین فازی (زبان طبیعی):
  1. اگر probability بالا باشد  -> قطعاً نگه‌دار (بدون توجه به centrality)
  2. اگر probability متوسط و centrality بالا باشد -> نگه‌دار (یال مهم است)
  3. اگر probability متوسط و centrality پایین باشد -> هرس کن
  4. اگر probability پایین و centrality بالا باشد -> نگه‌دار (احتیاط)
  5. اگر probability پایین و centrality پایین باشد -> قطعاً هرس کن

این رویکرد از یک آستانه‌ی سخت تک‌بعدی هوشمندتر است، چون یال‌هایی که
مدل به آن‌ها مطمئن نیست (probability متوسط) را بر اساس اهمیت ساختاری
واقعی‌شان (centrality) قضاوت می‌کند، نه فقط یک عدد.

بدون نیاز به کتابخانه‌ی خارجی (scikit-fuzzy) — همه‌چیز با numpy خام
پیاده‌سازی شده تا هیچ وابستگی جدیدی لازم نباشد.

نحوه‌ی اجرا:
    python benchmarks/fuzzy_pruning.py --city Eindhoven --seeds 5
    python benchmarks/fuzzy_pruning.py --city Rome --seeds 3
"""

import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
import osmnx as ox
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import dijkstra, connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spanner_pipeline import (  # noqa: E402
    GeometricEdgeSAGE, glv_repair_directed, compute_edge_centrality, build_edge_features,
    CITIES, MC_SAMPLES, GLV_T_LIMIT, FINETUNE_LR, safe_weighted_adjacency
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_RUNS_DIR = PROJECT_ROOT / "results" / "raw_runs"
RAW_RUNS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_PATH = PROJECT_ROOT / "results" / "models" / "best_base_model.pt"
OUT_CSV = RAW_RUNS_DIR / f"fuzzy_pruning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════
# توابع عضویت فازی مثلثی (Triangular Membership Functions)
# ══════════════════════════════════════════════════════════════
def tri_mf(x, a, b, c):
    """تابع عضویت مثلثی استاندارد: صفر در a، یک در b، صفر در c."""
    x = np.asarray(x, dtype=float)
    left = (x - a) / (b - a + 1e-9)
    right = (c - x) / (c - b + 1e-9)
    return np.clip(np.minimum(left, right), 0, 1)


def fuzzy_low(x):
    return tri_mf(x, -0.2, 0.0, 0.5)


def fuzzy_medium(x):
    return tri_mf(x, 0.15, 0.45, 0.75)


def fuzzy_high(x):
    return tri_mf(x, 0.5, 1.0, 1.2)


def fuzzy_pruning_score(probability, centrality_norm):
    """
    خروجی: یک عدد بین ۰ (قطعاً هرس کن) تا ۱ (قطعاً نگه‌دار)، با ترکیب
    ۵ قانون فازی بالا با روش Mamdani + defuzzification (centroid ساده).
    probability و centrality_norm باید بین ۰ و ۱ نرمال شده باشند.
    """
    p_low, p_med, p_high = fuzzy_low(probability), fuzzy_medium(probability), fuzzy_high(probability)
    c_low, c_med, c_high = fuzzy_low(centrality_norm), fuzzy_medium(centrality_norm), fuzzy_high(centrality_norm)

    rule1 = p_high
    rule2 = np.minimum(p_med, c_high)
    rule3 = np.minimum(p_med, c_low)
    rule4 = np.minimum(p_low, c_high)
    rule5 = np.minimum(p_low, c_low)

    numerator = (rule1 * 1.0 + rule2 * 1.0 + rule3 * 0.0 + rule4 * 0.7 + rule5 * 0.0)
    denominator = (rule1 + rule2 + rule3 + rule4 + rule5 + 1e-9)
    return numerator / denominator


def load_city_scc(city_query):
    G_raw = ox.graph_from_place(city_query, network_type="drive")
    nodes_raw = list(G_raw.nodes())
    adj_raw = nx.adjacency_matrix(G_raw, nodelist=nodes_raw, weight="length")
    _, labels = connected_components(adj_raw, directed=True, connection="strong")
    largest_cc_idx = np.argmax(np.bincount(labels))
    nodes_to_keep = [nodes_raw[i] for i in range(len(nodes_raw)) if labels[i] == largest_cc_idx]
    return G_raw.subgraph(nodes_to_keep).copy()


def prepare_features(G):
    log("  Computing PageRank + Edge Betweenness Centrality...")
    pagerank = nx.pagerank(G, weight="length")
    edge_list = list(G.edges(data=True))
    node_map = {n: i for i, n in enumerate(G.nodes())}
    raw_node = np.array([[G.in_degree(n), G.out_degree(n), pagerank.get(n, 1e-4)] for n in G.nodes()])
    x_local = torch.tensor(StandardScaler().fit_transform(raw_node), dtype=torch.float)
    edge_idx = torch.tensor(
        [[node_map[u] for u, v, d in edge_list], [node_map[v] for u, v, d in edge_list]],
        dtype=torch.long,
    )
    centrality_dict = compute_edge_centrality(G, weight="length")
    edge_attr_loc = build_edge_features(edge_list, centrality_dict)
    raw_centrality = np.array([centrality_dict.get((u, v), 0.0) for (u, v, d) in edge_list])
    if raw_centrality.max() > raw_centrality.min():
        centrality_01 = (raw_centrality - raw_centrality.min()) / (raw_centrality.max() - raw_centrality.min())
    else:
        centrality_01 = np.zeros_like(raw_centrality)
    return edge_list, x_local, edge_idx, edge_attr_loc, centrality_01


def compute_stretch_and_scc(G_orig, G_final, seed=42, num_sources=500):
    nodes = list(G_orig.nodes())
    matrix_orig = safe_weighted_adjacency(G_orig, nodelist=nodes, weight="length")
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
    MIN_SAFE_DEGREE = 2
    live_out_degree = dict(G.out_degree())
    live_in_degree = dict(G.in_degree())
    for i, (u, v, d) in enumerate(edge_list):
        is_bottleneck = (live_in_degree[v] <= MIN_SAFE_DEGREE) or (live_out_degree[u] <= MIN_SAFE_DEGREE)
        if keep_mask[i] or is_bottleneck:
            G_sparse.add_edge(u, v, length=d["length"])
        else:
            removed.append({"u": u, "v": v, "length": d["length"], "idx": i})
            live_out_degree[u] -= 1
            live_in_degree[v] -= 1
    G_final, repairs = glv_repair_directed(G_sparse, removed, t_limit=t_limit)
    return G_final, removed, repairs


def get_calibrated_probs(model, x_local, edge_idx, edge_attr_loc, seed):
    torch.manual_seed(seed)
    model.train()
    mc_preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            mc_preds.append(torch.sigmoid(model(x_local, edge_idx, edge_attr_loc)).numpy().flatten())
    mean_probs = np.mean(mc_preds, axis=0)
    std_probs = np.std(mc_preds, axis=0)
    return mean_probs + 0.5 * std_probs


def build_matched_random_mask(G, edge_list, n_removed_target, seed):
    n_edges = len(edge_list)
    non_bottleneck_idx = [i for i, (u, v, d) in enumerate(edge_list)
                           if not ((G.in_degree(v) <= 1) or (G.out_degree(u) <= 1))]
    rng = np.random.default_rng(seed)
    n_removed_target = min(n_removed_target, len(non_bottleneck_idx))
    removed_idx = set(rng.choice(non_bottleneck_idx, size=n_removed_target, replace=False))
    keep_mask = np.ones(n_edges, dtype=bool)
    for i in removed_idx:
        keep_mask[i] = False
    return keep_mask


def mask_from_score_matched_count(score, n_edges, n_removed_target, edge_list, G):
    """
    کمترین score اول حذف می‌شود -> تطبیق دقیق تعداد حذف با سایر حالت‌ها.
    یال‌های bottleneck از ابتدا از فهرست کاندیدها کنار گذاشته می‌شوند
    (دقیقاً مثل build_matched_random_mask) تا اجبار به نگه‌داشتن آن‌ها
    باعث انحراف sparsification واقعی از هدف نشود.
    """
    non_bottleneck_idx = np.array([i for i, (u, v, d) in enumerate(edge_list)
                                    if not ((G.in_degree(v) <= 1) or (G.out_degree(u) <= 1))])
    scores_nb = score[non_bottleneck_idx]
    order = np.argsort(scores_nb)
    n_removed_target = min(n_removed_target, len(non_bottleneck_idx))
    remove_idx = set(non_bottleneck_idx[order[:n_removed_target]])

    keep_mask = np.ones(n_edges, dtype=bool)
    for i in remove_idx:
        keep_mask[i] = False
    return keep_mask


def active_feedback_finetune(model, x_local, edge_idx, edge_attr_loc, edge_list,
                              removed_edges, repaired_indices, n_edges, epochs=5, lr=FINETUNE_LR):
    model_ft = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=lr)
    repaired_set = set(repaired_indices)
    removed_idx_set = {r["idx"] for r in removed_edges}
    y_vals = [1.0 if (i in repaired_set or i not in removed_idx_set) else 0.0 for i in range(n_edges)]
    y_feedback = torch.tensor(y_vals, dtype=torch.float).view(-1, 1)
    model_ft.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        out = model_ft(x_local, edge_idx, edge_attr_loc)
        loss = F.binary_cross_entropy_with_logits(out, y_feedback)
        loss.backward()
        optimizer.step()
    return model_ft


def _row(city, variant, seed, spars_pct, stretches, repairs, scc):
    return {
        "city": city, "variant": variant, "seed": seed,
        "sparsification_pct": spars_pct,
        "max_stretch": float(np.max(stretches)) if len(stretches) else float("nan"),
        "n_repairs": len(repairs),
        "scc_connected": scc == 1,
    }


def run_city(city_label, city_query, base_model, n_seeds, base_threshold=0.40):
    log(f"\n{'='*60}\n{city_label}\n{'='*60}")
    G = load_city_scc(city_query)
    edge_list, x_local, edge_idx, edge_attr_loc, centrality_01 = prepare_features(G)
    n_edges = len(edge_list)
    log(f"{G.number_of_nodes()} nodes, {n_edges} edges")

    rows = []
    for seed in range(n_seeds):
        calibrated = get_calibrated_probs(base_model, x_local, edge_idx, edge_attr_loc, seed)

        mask_A = calibrated > base_threshold
        _, removed_A, _ = build_and_repair(G, edge_list, mask_A)
        n_target = len(removed_A)

        # --- E) Fuzzy Pruning Decision ---
        fuzzy_score = fuzzy_pruning_score(calibrated, centrality_01)
        mask_E = mask_from_score_matched_count(fuzzy_score, n_edges, n_target, edge_list, G)
        G_final_E, removed_E, repairs_E = build_and_repair(G, edge_list, mask_E)
        stretches_E, scc_E = compute_stretch_and_scc(G, G_final_E, seed=seed)
        rows.append(_row(city_label, "E_Fuzzy_Pruning", seed,
                          len(removed_E)/n_edges*100, stretches_E, repairs_E, scc_E))

        # --- C) GA-threshold + Active Feedback ---
        mask_C0 = calibrated > base_threshold
        _, removed_C0, repairs_C0 = build_and_repair(G, edge_list, mask_C0)
        model_ft = active_feedback_finetune(base_model, x_local, edge_idx, edge_attr_loc,
                                             edge_list, removed_C0, repairs_C0, n_edges)
        calibrated_C = get_calibrated_probs(model_ft, x_local, edge_idx, edge_attr_loc, seed)
        mask_C = mask_from_score_matched_count(calibrated_C, n_edges, n_target, edge_list, G)
        G_final_C, removed_C, repairs_C = build_and_repair(G, edge_list, mask_C)
        stretches_C, scc_C = compute_stretch_and_scc(G, G_final_C, seed=seed)
        rows.append(_row(city_label, "C_GA_Threshold_plus_Feedback", seed,
                          len(removed_C)/n_edges*100, stretches_C, repairs_C, scc_C))

        # --- D) Random Pruning (هم‌سطح دقیق) ---
        mask_D = build_matched_random_mask(G, edge_list, n_target, seed)
        G_final_D, removed_D, repairs_D = build_and_repair(G, edge_list, mask_D)
        stretches_D, scc_D = compute_stretch_and_scc(G, G_final_D, seed=seed)
        rows.append(_row(city_label, "D_Random_Pruning(matched)", seed,
                          len(removed_D)/n_edges*100, stretches_D, repairs_D, scc_D))

        log(f"  seed={seed} | target_removed={n_target} | "
            f"Fuzzy(E)={repairs_E} | GA+Feedback(C)={repairs_C} | Random(D)={repairs_D}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="Eindhoven")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    model = GeometricEdgeSAGE()
    model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True, map_location="cpu"))

    rows = run_city(args.city, CITIES[args.city], model, args.seeds)

    df = pd.DataFrame(rows)
    write_header = not OUT_CSV.exists()
    df.to_csv(OUT_CSV, mode="a", header=write_header, index=False)

    print("\n" + "=" * 70)
    print("خلاصه‌ی میانگین هر Variant (E=Fuzzy, C=GA+Feedback, D=Random):")
    print(df.groupby("variant")[["sparsification_pct", "max_stretch", "n_repairs"]].mean().to_string())
    print("=" * 70)
    print(f"\nذخیره شد در: {OUT_CSV}")


if __name__ == "__main__":
    main()