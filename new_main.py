
def main():
    WEIGHTS = 'best_base_model.pt'

    # 1. Run base pipeline and collect graph representations
    print("Step 1: Running base GNN model pipeline...")
    model = train_base_model(weights_path=WEIGHTS)

    worker_args = [(lbl, qry, WEIGHTS) for lbl, qry in CITIES.items()]
    city_results = []

    print("\nStep 2: Processing cities in parallel to collect graph representations...")
    with ProcessPoolExecutor(max_workers=len(CITIES)) as pool:
        futures = {pool.submit(city_worker, arg): arg[0] for arg in worker_args}
        for future in as_completed(futures):
            lbl = futures[future]
            try:
                city_results.append(future.result())
                print(f"[Main Pipeline] \u2713 {lbl} loading completed.")
            except Exception as exc:
                print(f"[Main Pipeline] \u2717 {lbl} failed: {exc}")

    # 2. Fine-tune on repaired edges
    print("\nStep 3: Fine-tuning GNN on repaired edges (Phase 2)...")
    model = finetune_on_repairs_continual(model, city_results)

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

        G_opt = nx.Graph()
        G_opt.add_nodes_from(G.nodes())
        removed = []
        for i, (u, v, k, d) in enumerate(edge_list):
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
