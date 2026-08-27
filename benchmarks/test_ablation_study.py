"""
ABLATION STUDY: measuring the impact of each pipeline component.
==================================================================
This script runs 4 versions of the algorithm to justify the use of:
1. MC-Dropout (Uncertainty)
2. Replay Buffer (Continual Learning)
3. SCC Extraction (Directed Logic)
"""

import torch
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIGURES_DIR = REPO_ROOT / "results" / "figures"
DEFAULT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# وارد کردن تمام اجزای ابر-پروژه شما
from spanner_pipeline import (
    GeometricEdgeSAGE, city_worker, train_base_model, 
    finetune_on_repairs_continual, compute_final_metrics_directed, CITIES,
    DEFAULT_MODEL_PT
)

def run_ablation_experiment():
    print("--- 🔬 Starting Ablation Study (City: Eindhoven) ---")
    
    city_key = "Eindhoven"
    city_query = CITIES[city_key]
    weights_path = DEFAULT_MODEL_PT
    
    # آموزش مدل پایه برای شروع
    model_base = train_base_model(weights_path=weights_path)
    
    ablation_results = []

    # سناریو ۱: مدل کامل (The Full Masterpiece)
    print("\n[Scenario 1] Running Full Proposed Framework...")
    # ما از همان توابع اصلی استفاده می‌کنیم چون آن‌ها کامل هستند
    args = (city_key, city_query, weights_path)
    res_full = city_worker(args)
    # فین‌تیونینگ با حافظه (Replay Buffer فعلا در همین یک شهر شبیه‌سازی می‌شود)
    model_full, _ = finetune_on_repairs_continual(model_base, [res_full])
    metrics_full = compute_final_metrics_directed(model_full, res_full)
    ablation_results.append({
        "Configuration": "Full Framework (Proposed)",
        "Sparsification": metrics_full["Sparsification"],
        "Max Stretch": metrics_full["Max Stretch"],
        "SCC": metrics_full["SCC"]
    })

    # سناریو ۲: بدون عدم قطعیت (No Uncertainty)
    print("\n[Scenario 2] Running without MC-Dropout...")
    # در این حالت ما MC_SAMPLES را در ذهن مدل ۱ فرض می‌کنیم (تغییر در زمان اجرا)
    import spanner_pipeline
    original_mc = spanner_pipeline.MC_SAMPLES
    spanner_pipeline.MC_SAMPLES = 1 # خاموش کردن نمونه‌گیری آماری
    
    res_no_unc = city_worker(args)
    metrics_no_unc = compute_final_metrics_directed(model_full, res_no_unc)
    ablation_results.append({
        "Configuration": "w/o MC-Dropout (Gap 9)",
        "Sparsification": metrics_no_unc["Sparsification"],
        "Max Stretch": metrics_no_unc["Max Stretch"],
        "SCC": metrics_no_unc["SCC"]
    })
    spanner_pipeline.MC_SAMPLES = original_mc # بازگرداندن به حالت قبل

    # سناریو ۳: بدون محافظت گره‌ها (No Safeguards) — اجرای واقعی
    print("\n[Scenario 3] Running WITHOUT topology safeguard (is_bottleneck disabled)...")
    metrics_no_safeguard = compute_final_metrics_directed(model_full, res_full, disable_safeguard=True)
    ablation_results.append({
        "Configuration": "w/o Topology Safeguards",
        "Sparsification": metrics_no_safeguard["Sparsification"],
        "Max Stretch": metrics_no_safeguard["Max Stretch"],
        "SCC": metrics_no_safeguard["SCC"]
    })

    # نمایش جدول نهایی ابطالی
    print("\n" + "="*85)
    print("                FINAL ABLATION STUDY RESULTS")
    print("="*85)
    df = pd.DataFrame(ablation_results)
    print(df.to_string(index=False))
    print("="*85)
    
    # رسم نمودار مقایسه‌ای ابطالی برای مقاله
    plt.figure(figsize=(10, 5))
    configs = [r["Configuration"] for r in ablation_results[:2]]
    stretches = [float(r["Max Stretch"]) for r in ablation_results[:2]]
    
    plt.bar(configs, stretches, color=['green', 'red'], alpha=0.7)
    plt.axhline(1.5, color='black', linestyle='--', label='Theoretical Limit')
    plt.title("Ablation Analysis: Impact of MC-Dropout on Max Stretch", fontweight='bold')
    plt.ylabel("Maximum Stretch Factor")
    plt.legend()
    plt.savefig(DEFAULT_FIGURES_DIR / "ablation_analysis.png", dpi=300)
    print(f"\n✓ Ablation Plot saved to '{DEFAULT_FIGURES_DIR / 'ablation_analysis.png'}'")

if __name__ == "__main__":
    run_ablation_experiment()