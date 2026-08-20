import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spanner_pipeline import (
    GeometricEdgeSAGE, train_base_model, city_worker,
    finetune_on_repairs_continual, glv_repair_directed, CITIES,
    DEFAULT_MODEL_PT, DEFAULT_FIGURES_DIR
)
from monte_carlo_validator import run_monte_carlo_scipy


def evaluate_city_with_model(model, res):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            model(res["_x_local"], res["_edge_idx"], res["_edge_attr_loc"])
        ).numpy().flatten()

    G = res["_G"]
    edge_list = res["_edge_list"]
    G_opt = nx.DiGraph()
    G_opt.add_nodes_from(G.nodes())
    removed = []
    for i, (u, v, d) in enumerate(edge_list):
        if probs[i] > 0.55:
            G_opt.add_edge(u, v, length=d["length"])
        else:
            removed.append({'u': u, 'v': v, 'length': d['length'], 'prob': probs[i], 'idx': i})

    G_final, _ = glv_repair_directed(G_opt, removed)
    stretches = run_monte_carlo_scipy(res["city_label"], G, G_final, num_samples=100000)
    return stretches


def main():
    WEIGHTS = DEFAULT_MODEL_PT

    print("\n[LOCO] Training base model (shared starting point for all folds)...")
    train_base_model(weights_path=WEIGHTS)

    print("\n[LOCO] Collecting graph representations for all cities...")
    city_results = []
    for lbl, qry in CITIES.items():
        city_results.append(city_worker((lbl, qry, WEIGHTS)))
        print(f"[LOCO] \u2713 {lbl} loaded.")

    results_by_city = {r["city_label"]: r for r in city_results}

    fold_stats = []
    per_city_stretches = {}

    for held_out_label in CITIES.keys():
        print(f"\n{'='*70}")
        print(f"[LOCO] FOLD: held-out city = {held_out_label}")
        print(f"{'='*70}")

        train_results = [r for r in city_results if r["city_label"] != held_out_label]
        held_out_res = results_by_city[held_out_label]

        model = GeometricEdgeSAGE()
        model.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location="cpu"))

        model, _ = finetune_on_repairs_continual(model, train_results)

        print(f"[LOCO] Evaluating zero-shot on held-out city: {held_out_label}...")
        stretches = evaluate_city_with_model(model, held_out_res)
        per_city_stretches[held_out_label] = stretches

        mean_s = np.mean(stretches)
        median_s = np.median(stretches)
        p99_s = np.percentile(stretches, 99)
        max_s = np.max(stretches)
        violation_rate = np.mean(stretches > 1.5) * 100

        fold_stats.append({
            'Held-out City': held_out_label,
            'Trained on': ", ".join(r["city_label"] for r in train_results),
            'Mean Stretch': f"{mean_s:.4f}",
            'Median Stretch': f"{median_s:.4f}",
            '99th Percentile': f"{p99_s:.4f}",
            'Max Stretch': f"{max_s:.4f}",
            'Violation Rate (>1.5)': f"{violation_rate:.4f}%"
        })

    print("\n" + "="*90)
    print("     LEAVE-ONE-CITY-OUT CROSS-VALIDATION RESULTS (ZERO-SHOT GENERALIZATION)")
    print("="*90)
    df = pd.DataFrame(fold_stats)
    print(df.to_string(index=False))
    print("="*90)

    df.to_csv(DEFAULT_FIGURES_DIR.parent / "raw_runs" / "leave_one_city_out_results.csv", index=False)
    print(f"\n[LOCO] \u2713 Results saved to results/raw_runs/leave_one_city_out_results.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    for ax, (city_label, stretches) in zip(axes_flat, per_city_stretches.items()):
        sorted_data = np.sort(stretches)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        mean_s = np.mean(stretches)
        max_s = np.max(stretches)
        ax.plot(sorted_data, yvals, linewidth=2.5, color='#d62728')
        ax.axvline(x=1.5, color='red', linestyle='--', linewidth=1.5, label='Limit (t=1.5)')
        ax.set_title(f"{city_label} (held out, zero-shot)", fontsize=12, fontweight='bold')
        ax.set_xlabel(r'Stretch Ratio $d_{H}/d_{G}$', fontsize=10)
        ax.set_ylabel(r'P(Stretch $\leq$ x)', fontsize=10)
        ax.grid(True, alpha=0.4)
        ax.legend(loc='lower right', fontsize=9)
        data_max = sorted_data.max()
        ax.set_xlim(sorted_data.min() * 0.995, max(data_max * 1.02, 1.52))
        ax.text(0.03, 0.7, f"Mean: {mean_s:.3f}\nMax: {max_s:.3f}", transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Leave-One-City-Out: Zero-Shot Generalization to Unseen Topology', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(DEFAULT_FIGURES_DIR / 'leave_one_city_out_cdf.png', dpi=300)
    print("[LOCO] \u2713 Plot saved to results/figures/leave_one_city_out_cdf.png")


if __name__ == '__main__':
    main()
