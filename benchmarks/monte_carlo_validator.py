import os
import time
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# وارد کردن توابع مورد نیاز از فایل اصلی پروژه
from spanner_pipeline import (
    DEFAULT_FIGURES_DIR,
    GeometricEdgeSAGE, train_base_model, city_worker,
    finetune_on_repairs_continual, glv_repair_directed, CITIES
)

def run_monte_carlo_scipy(city_label, G_orig, G_final, num_samples=100000):
    """محاسبه فوق‌سریع ضریب کشش با استفاده از الگوریتم دایجسترا در SciPy (کدهای C++)"""
    print(f"\n[MC - {city_label}] Starting ultra-fast SciPy Monte Carlo ({num_samples} pairs)...")
    t0 = time.time()
    
    # نگاشت گره‌ها به ایندکس‌های عددی از 0 تا N-1
    nodes = list(G_orig.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    # تبدیل گراف‌های NetworkX به ماتریس اسپرس در SciPy
    matrix_orig = nx.adjacency_matrix(G_orig, nodelist=nodes, weight='length')
    matrix_final = nx.adjacency_matrix(G_final, nodelist=nodes, weight='length')
    
    # انتخاب تصادفی گره‌های مبدا
    rng = np.random.default_rng(42)
    num_sources = min(num_nodes, 1000)  # انتخاب حداکثر 1000 مبدا تصادفی
    sources = rng.choice(num_nodes, size=num_sources, replace=False)
    
    # اجرای محاسبات دایجسترا با کدهای کامپایل‌شده C++ در SciPy
    print(f"[MC - {city_label}] Running SciPy parallelized Dijkstra for {num_sources} source nodes...")
    dist_orig = dijkstra(matrix_orig, directed=True, indices=sources)
    dist_span = dijkstra(matrix_final, directed=True, indices=sources)
    
    # استخراج زوج گره‌های معتبر و محاسبه ضریب کشش
    stretches = []
    for i in range(num_sources):
        d_orig_row = dist_orig[i]
        d_span_row = dist_span[i]
        
        # فیلتر کردن مقادیر نامعتبر (فاصله صفر یا بی‌نهایت)
        valid_mask = (d_orig_row > 0) & (d_orig_row < np.inf) & (d_span_row < np.inf)
        if not np.any(valid_mask):
            continue
            
        ratios = d_span_row[valid_mask] / d_orig_row[valid_mask]
        stretches.extend(ratios.tolist())
        
    # نمونه‌برداری نهایی برای رسیدن به ۱۰۰,۰۰۰ نمونه
    if len(stretches) > num_samples:
        stretches = rng.choice(stretches, size=num_samples, replace=False)
        
    elapsed = time.time() - t0
    print(f"[MC - {city_label}] Finished processing {len(stretches)} routes in {elapsed:.2f}s!")
    return np.array(stretches)

def main():
    WEIGHTS = 'best_base_model.pt'

    # 1. Run base pipeline and collect graph representations
    print("Step 1: Running base GNN model pipeline...")
    model = train_base_model(weights_path=WEIGHTS)

    worker_args = [(lbl, qry, WEIGHTS) for lbl, qry in CITIES.items()]
    city_results = []

    print("\nStep 2: Processing cities in parallel to collect graph representations...")
    for arg in worker_args:
        lbl = arg[0]
        try:
            city_results.append(city_worker(arg))
            print(f"[Main Pipeline] processing {lbl} completed.")
        except Exception as exc:
            print(f"[Main Pipeline] {lbl} failed: {exc}")
    # 2. Fine-tune on repaired edges
    print("\nStep 3: Fine-tuning GNN on repaired edges (Phase 2)...")
    model, _ = finetune_on_repairs_continual(model, city_results)

    # 3. Independent Monte Carlo verification for ALL cities (no city restriction)
    print("\nStep 4: Running Monte Carlo Verification (all cities)...")

    stats_summary = []
    per_city_stretches = {}

    for res in city_results:
        city_label = res['city_label']

        G = res['_G']
        edge_list = res['_edge_list']
        x_local = res['_x_local']
        edge_idx = res['_edge_idx']
        edge_attr = res['_edge_attr_loc']

        model.eval()
        with torch.no_grad():
            final_probs = torch.sigmoid(model(x_local, edge_idx, edge_attr)).numpy().flatten()

        G_opt = nx.DiGraph()
        G_opt.add_nodes_from(G.nodes())
        removed = []
        for i, (u, v, d) in enumerate(edge_list):
            if final_probs[i] > 0.55:
                G_opt.add_edge(u, v, length=d['length'])
            else:
                removed.append({'u': u, 'v': v, 'length': d['length'], 'prob': final_probs[i], 'idx': i})

        G_final, _ = glv_repair_directed(G_opt, removed)

        stretches = run_monte_carlo_scipy(city_label, G, G_final, num_samples=100000)
        per_city_stretches[city_label] = stretches

        mean_s = np.mean(stretches)
        median_s = np.median(stretches)
        p99_s = np.percentile(stretches, 99)
        max_s = np.max(stretches)
        violation_rate = np.mean(stretches > 1.5) * 100

        stats_summary.append({
            'City': city_label,
            'Mean Stretch': f"{mean_s:.4f}",
            'Median Stretch': f"{median_s:.4f}",
            '99th Percentile': f"{p99_s:.4f}",
            'Max Stretch': f"{max_s:.4f}",
            'Violation Rate (>1.5)': f"{violation_rate:.4f}%"
        })

    # ── Plot 1: original combined overlay (kept, path fixed to results/figures/) ──
    plt.figure(figsize=(10, 6))
    for city_label, stretches in per_city_stretches.items():
        sorted_data = np.sort(stretches)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        max_s = np.max(stretches)
        plt.plot(sorted_data, yvals, label=f'{city_label} (Max Stretch: {max_s:.3f})', linewidth=2.5)
    plt.axvline(x=1.5, color='red', linestyle='--', label=r'Theoretical Limit (t = 1.5)', linewidth=1.5)
    plt.title(r'Empirical Cumulative Distribution Function (CDF) of Global Stretch', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(r'Global Stretch Ratio $d_{G_{final}}(u,v) / d_G(u,v)$', fontsize=12)
    plt.ylabel(r'P(Stretch $\leq$ x)', fontsize=12)
    plt.xlim(0.98, 1.6)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(DEFAULT_FIGURES_DIR / 'global_stretch_cdf.png', dpi=300)
    print("\n[Main] \u2713 Combined overlay plot saved to results/figures/global_stretch_cdf.png")
    plt.close()

    # ── Plot 2: NEW 2x2 subplot grid, one axis per city, self-contained annotations ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for ax, (city_label, stretches) in zip(axes_flat, per_city_stretches.items()):
        sorted_data = np.sort(stretches)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        mean_s = np.mean(stretches)
        median_s = np.median(stretches)
        max_s = np.max(stretches)

        ax.plot(sorted_data, yvals, linewidth=2.5, color='#1f77b4')
        ax.axvline(x=1.5, color='red', linestyle='--', linewidth=1.5, label='Limit (t=1.5)')
        ax.set_title(city_label, fontsize=13, fontweight='bold')
        ax.set_xlabel(r'Stretch Ratio $d_{H}/d_{G}$', fontsize=10)
        ax.set_ylabel(r'P(Stretch $\leq$ x)', fontsize=10)
        ax.grid(True, alpha=0.4)
        ax.legend(loc='lower right', fontsize=9)

        # Scale x-axis to this city's own data range (with margin), not a fixed shared range
        data_max = sorted_data.max()
        ax.set_xlim(sorted_data.min() * 0.995, max(data_max * 1.02, 1.52))

        annotation = f"Mean: {mean_s:.3f}\nMedian: {median_s:.3f}\nMax: {max_s:.3f}"
        ax.text(0.03, 0.7, annotation, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide any unused subplot axes if fewer than 4 cities were processed
    for ax in axes_flat[len(per_city_stretches):]:
        ax.set_visible(False)

    fig.suptitle('Empirical CDF of Directed Global Stretch, by City', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(DEFAULT_FIGURES_DIR / 'stretch_cdf_by_city.png', dpi=300)
    print("[Main] \u2713 Per-city subplot grid saved to results/figures/stretch_cdf_by_city.png")
    plt.close()

    # Print statistics table
    print("\n" + "="*85)
    print("     MONTE CARLO STRETCH VALIDATION METRICS (10^5 RANDOM ROUTE PAIRS, ALL CITIES)")
    print("="*85)
    print(pd.DataFrame(stats_summary).to_string(index=False))
    print("="*85)


if __name__ == '__main__':
    main()
