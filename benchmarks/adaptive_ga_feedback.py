"""
adaptive_ga_feedback.py (نسخه‌ی ۲ — با قید سخت + اثبات برتری GA)
====================================================================
اصلاحات نسبت به نسخه‌ی قبلی:

۱) قید سخت روی sparsification: راه‌حل‌هایی که بیش از ±3 درصد از هدف
   فاصله دارن، به‌شدت جریمه می‌شن (غیرقابل‌قبول اعلام می‌شن)، نه فقط
   جریمه‌ی ملایم. این از "راه‌حل قهرمان دروغین" (هرس نکردن) جلوگیری
   می‌کند.

۲) اضافه‌شدن RANDOM SEARCH با همان تعداد ارزیابی (budget) که GA دارد.
   اگر GA بتواند با همان تعداد فراخوانی تابع فیتنس، به نتیجه‌ی بهتری
   برسد، این مستقیماً ثابت می‌کند که ساختار تکاملی (انتخاب + تولید نسل)
   واقعاً کمک می‌کند — نه اینکه GA فقط "شانسی" بهتر بوده.

نحوه‌ی اجرا:
    python benchmarks/adaptive_ga_feedback.py --city Paris --seeds 3
    python benchmarks/adaptive_ga_feedback.py --city Rome --seeds 2
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
OUT_CSV = RAW_RUNS_DIR / f"adaptive_ga_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

SPARSIFICATION_TOLERANCE_PCT = 3.0  # حداکثر فاصله‌ی قابل‌قبول از هدف


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_city_scc(city_query):
    G_raw = ox.graph_from_place(city_query, network_type="drive")
    nodes_raw = list(G_raw.nodes())
    adj_raw = nx.adjacency_matrix(G_raw, nodelist=nodes_raw, weight="length")
    _, labels = connected_components(adj_raw, directed=True, connection="strong")
    largest_cc_idx = np.argmax(np.bincount(labels))
    nodes_to_keep = [nodes_raw[i] for i in range(len(nodes_raw)) if labels[i] == largest_cc_idx]
    return G_raw.subgraph(nodes_to_keep).copy()


def prepare_features(G):
    log("  Computing PageRank + Edge Betweenness Centrality (once per city)...")
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
    return edge_list, x_local, edge_idx, edge_attr_loc


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


# ──────────────────────────────────────────────
# تابع فیتنس مشترک (با قید سخت) — هم GA هم Random Search از این استفاده می‌کنن
# ──────────────────────────────────────────────
def make_fitness_fn(calibrated, G, edge_list, target_spars_pct, tolerance=SPARSIFICATION_TOLERANCE_PCT):
    n_edges = len(edge_list)

    def fitness(threshold):
        keep_mask = calibrated > threshold
        _, removed, repairs = build_and_repair(G, edge_list, keep_mask)
        achieved_spars = len(removed) / n_edges * 100
        diff = abs(achieved_spars - target_spars_pct)

        if diff > tolerance:
            # غیرقابل‌قبول: جریمه‌ی بسیار بزرگ و متناسب با میزان تخلف
            # (بزرگ‌تر از هر مقدار ممکن برای n_repairs، تا GA/RS همیشه
            # راه‌حل feasible را به غیرfeasible ترجیح دهد)
            score = -1_000_000 - diff * 10_000
        else:
            score = -len(repairs)  # فقط بین راه‌حل‌های feasible مقایسه کن

        return score, len(repairs), achieved_spars

    return fitness


# ──────────────────────────────────────────────
# ۱. GENETIC ALGORITHM
# ──────────────────────────────────────────────
def ga_optimize_threshold(fitness_fn, seed=0, pop_size=8, n_generations=6):
    rng = np.random.default_rng(seed)
    population = rng.uniform(0.15, 0.65, size=pop_size)

    n_evals = 0
    best_threshold, best_score = None, -np.inf
    history = []

    for gen in range(n_generations):
        scored = [(t,) + fitness_fn(t) for t in population]
        n_evals += len(population)
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored[0][1] > best_score:
            best_score = scored[0][1]
            best_threshold = scored[0][0]
        history.append(best_score)
        log(f"  [GA] gen={gen} | best_threshold={scored[0][0]:.3f} | repairs={scored[0][2]} | "
            f"spars={scored[0][3]:.2f}% | feasible={scored[0][1] > -1_000_000}")

        survivors = [s[0] for s in scored[:3]]  # الیتیسم: ۳ نفر برتر
        new_pop = list(survivors)
        while len(new_pop) < pop_size:
            parent = rng.choice(survivors)
            child = np.clip(parent + rng.normal(0, 0.05), 0.02, 0.80)
            new_pop.append(child)
        population = np.array(new_pop)

    return best_threshold, best_score, n_evals, history


# ──────────────────────────────────────────────
# ۲. RANDOM SEARCH — با همان budget (تعداد ارزیابی) که GA داشت
#    این برای اثبات "GA دلخواه نبوده" لازم است
# ──────────────────────────────────────────────
def random_search_optimize_threshold(fitness_fn, n_evals, seed=100):
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(0.02, 0.80, size=n_evals)
    best_threshold, best_score = None, -np.inf
    history = []
    for i, t in enumerate(candidates):
        score, repairs, spars = fitness_fn(t)
        if score > best_score:
            best_score = score
            best_threshold = t
        history.append(best_score)
    log(f"  [RandomSearch] best_threshold={best_threshold:.3f} | best_score={best_score:.1f} "
        f"(with same budget={n_evals} evaluations as GA)")
    return best_threshold, best_score, n_evals, history


# ──────────────────────────────────────────────
# ۳. ACTIVE SELF-CORRECTING FEEDBACK LOOP
# ──────────────────────────────────────────────
def active_feedback_finetune(model, x_local, edge_idx, edge_attr_loc, edge_list,
                              removed_edges, repaired_indices, n_edges, epochs=5, lr=FINETUNE_LR):
    model_ft = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=lr)

    repaired_set = set(repaired_indices)
    removed_idx_set = {r["idx"] for r in removed_edges}

    y_vals = []
    for i in range(n_edges):
        if i in repaired_set:
            y_vals.append(1.0)
        elif i in removed_idx_set:
            y_vals.append(0.0)
        else:
            y_vals.append(1.0)
    y_feedback = torch.tensor(y_vals, dtype=torch.float).view(-1, 1)

    log(f"  [Feedback] Fine-tuning {epochs} epochs on {len(repaired_set)} corrected edges...")
    model_ft.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        out = model_ft(x_local, edge_idx, edge_attr_loc)
        loss = F.binary_cross_entropy_with_logits(out, y_feedback)
        loss.backward()
        optimizer.step()
    return model_ft


def build_matched_random_mask(G, edge_list, n_removed_target, seed):
    """
    دقیقاً n_removed_target یال را از بین یال‌های غیر-bottleneck به‌طور
    تصادفی برای حذف انتخاب می‌کند، با دنبال‌کردن درجه‌ی زنده تا هیچ
    گرهی زیر ۲ نیفتد (رفع همان باگ محافظ ثابت).
    """
    n_edges = len(edge_list)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_edges)

    MIN_SAFE_DEGREE = 2
    live_out_degree = dict(G.out_degree())
    live_in_degree = dict(G.in_degree())

    keep_mask = np.ones(n_edges, dtype=bool)
    n_removed = 0
    for i in order:
        if n_removed >= n_removed_target:
            break
        u, v, d = edge_list[i]
        is_bottleneck = (live_in_degree[v] <= MIN_SAFE_DEGREE) or (live_out_degree[u] <= MIN_SAFE_DEGREE)
        if not is_bottleneck:
            keep_mask[i] = False
            live_out_degree[u] -= 1
            live_in_degree[v] -= 1
            n_removed += 1
    return keep_mask


def _row(city, variant, seed, spars_pct, stretches, repairs, scc, extra=None):
    r = {
        "city": city, "variant": variant, "seed": seed,
        "sparsification_pct": spars_pct,
        "max_stretch": float(np.max(stretches)) if len(stretches) else float("nan"),
        "n_repairs": len(repairs),
        "scc_connected": scc == 1,
    }
    if extra:
        r.update(extra)
    return r


def run_city(city_label, city_query, base_model, n_seeds, base_threshold=0.40):
    log(f"\n{'='*60}\n{city_label}\n{'='*60}")
    G = load_city_scc(city_query)
    edge_list, x_local, edge_idx, edge_attr_loc = prepare_features(G)
    n_edges = len(edge_list)
    log(f"{G.number_of_nodes()} nodes, {n_edges} edges")

    rows = []
    for seed in range(n_seeds):
        # --- A) Base threshold (0.40) ---
        calibrated = get_calibrated_probs(base_model, x_local, edge_idx, edge_attr_loc, seed)
        mask_A = calibrated > base_threshold
        G_final_A, removed_A, repairs_A = build_and_repair(G, edge_list, mask_A)
        stretches_A, scc_A = compute_stretch_and_scc(G, G_final_A, seed=seed)
        target_spars = len(removed_A) / n_edges * 100
        rows.append(_row(city_label, "A_Base_Threshold_0.40", seed, target_spars,
                          stretches_A, repairs_A, scc_A))

        fitness_fn = make_fitness_fn(calibrated, G, edge_list, target_spars)

        # --- B1) GA ---
        best_t_ga, score_ga, n_evals_ga, hist_ga = ga_optimize_threshold(fitness_fn, seed=seed)
        mask_B1 = calibrated > best_t_ga
        G_final_B1, removed_B1, repairs_B1 = build_and_repair(G, edge_list, mask_B1)
        stretches_B1, scc_B1 = compute_stretch_and_scc(G, G_final_B1, seed=seed)
        rows.append(_row(city_label, f"B1_GA_Threshold({best_t_ga:.2f})", seed,
                          len(removed_B1)/n_edges*100, stretches_B1, repairs_B1, scc_B1,
                          extra={"n_evals": n_evals_ga}))

        # --- B2) Random Search با همان budget ---
        best_t_rs, score_rs, n_evals_rs, hist_rs = random_search_optimize_threshold(
            fitness_fn, n_evals=n_evals_ga, seed=seed + 100
        )
        mask_B2 = calibrated > best_t_rs
        G_final_B2, removed_B2, repairs_B2 = build_and_repair(G, edge_list, mask_B2)
        stretches_B2, scc_B2 = compute_stretch_and_scc(G, G_final_B2, seed=seed)
        rows.append(_row(city_label, f"B2_RandomSearch_Threshold({best_t_rs:.2f})", seed,
                          len(removed_B2)/n_edges*100, stretches_B2, repairs_B2, scc_B2,
                          extra={"n_evals": n_evals_rs}))

        # --- C) GA-threshold + Active Feedback Fine-tuning ---
        model_ft = active_feedback_finetune(base_model, x_local, edge_idx, edge_attr_loc,
                                             edge_list, removed_B1, repairs_B1, n_edges)
        calibrated_C = get_calibrated_probs(model_ft, x_local, edge_idx, edge_attr_loc, seed)
        mask_C = calibrated_C > best_t_ga
        G_final_C, removed_C, repairs_C = build_and_repair(G, edge_list, mask_C)
        stretches_C, scc_C = compute_stretch_and_scc(G, G_final_C, seed=seed)
        rows.append(_row(city_label, "C_GA_Threshold_plus_Feedback", seed,
                          len(removed_C)/n_edges*100, stretches_C, repairs_C, scc_C))

        # --- D) Random Pruning هم‌سطح با C (تطبیق دقیق تعداد، نه احتمال) ---
        mask_D = build_matched_random_mask(G, edge_list, len(removed_C), seed)
        G_final_D, removed_D, repairs_D = build_and_repair(G, edge_list, mask_D)
        stretches_D, scc_D = compute_stretch_and_scc(G, G_final_D, seed=seed)
        rows.append(_row(city_label, "D_Random_Pruning(matched_to_C)", seed,
                          len(removed_D)/n_edges*100, stretches_D, repairs_D, scc_D))

        log(f"  seed={seed} SUMMARY: A={repairs_A} | GA={repairs_B1} | RandSearch={repairs_B2} | "
            f"GA+Feedback={repairs_C} | RandomPrune={repairs_D}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="Paris")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    model = GeometricEdgeSAGE()
    model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True, map_location="cpu"))

    rows = run_city(args.city, CITIES[args.city], model, args.seeds)

    df = pd.DataFrame(rows)
    write_header = not OUT_CSV.exists()
    df.to_csv(OUT_CSV, mode="a", header=write_header, index=False)

    print("\n" + "=" * 70)
    print("خلاصه‌ی میانگین هر Variant:")
    print(df.groupby("variant")[["sparsification_pct", "max_stretch", "n_repairs"]].mean().to_string())
    print("=" * 70)
    print(f"\nذخیره شد در: {OUT_CSV}")


if __name__ == "__main__":
    main()