"""Strict leave-one-city-out evaluation.

For every held-out city and seed:
  1. the base model is trained on results/data/spanner_dataset_pro.csv WITHOUT the held-out city,
  2. the training cities are processed with that base model and its repair feedback is used for fine-tuning,
  3. the fine-tuned model is applied to the held-out city and 10^5 sampled pairs are measured.
Nothing derived from the held-out city enters training at any stage. Weights go to a temporary folder, so the
committed results/models/best_base_model.pt is never touched.

    python benchmarks/loco_strict.py                                  # 4 cities, 3 seeds (resumable)
    python benchmarks/loco_strict.py --seeds 5
    python benchmarks/loco_strict.py --cities Eindhoven Manhattan --seeds 1 --samples 5000 --out /tmp/loco_smoke.csv   # quick check
"""
import argparse
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch

from leave_one_city_out import evaluate_city_with_model
from loco_utils import STRICT_COLUMNS, pending_runs, stretch_summary
from spanner_pipeline import (CITIES, GeometricEdgeSAGE, city_worker, finetune_on_repairs_continual,
                              train_base_model)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(ROOT, "results", "raw_runs", "leave_one_city_out_strict.csv")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_fold(held_out, seed, cities, samples, workdir):
    weights = os.path.join(workdir, f"base_without_{held_out}_seed{seed}.pt")
    seed_all(seed)
    train_base_model(weights_path=weights, exclude_city=held_out)
    seed_all(seed)
    results = {lbl: city_worker((lbl, CITIES[lbl], weights)) for lbl in cities}
    train_results = [results[lbl] for lbl in cities if lbl != held_out]
    model = GeometricEdgeSAGE()
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    seed_all(seed)
    model, _ = finetune_on_repairs_continual(model, train_results)
    stretches = evaluate_city_with_model(model, results[held_out], num_samples=samples)
    row = {"seed": seed, "held_out": held_out, "trained_on": ", ".join(r["city_label"] for r in train_results),
           "base_excludes_held_out": True}
    row.update(stretch_summary(stretches))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=list(CITIES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=100000)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    if len(args.cities) < 2:
        ap.error("need at least two cities")

    done = pd.read_csv(args.out) if os.path.exists(args.out) else None
    todo = pending_runs(done, args.cities, range(args.seeds))
    print(f"[LOCO-strict] {len(todo)} run(s) to do, output: {args.out}")
    with tempfile.TemporaryDirectory() as workdir:
        for seed, held_out in todo:
            print(f"\n[LOCO-strict] seed={seed} held-out={held_out}")
            row = run_fold(held_out, seed, args.cities, args.samples, workdir)
            table = pd.DataFrame([row], columns=STRICT_COLUMNS)
            table.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
            print(f"[LOCO-strict] saved: mean={row['mean_stretch']:.4f} max={row['max_stretch']:.4f} "
                  f"violations={row['violation_rate_pct']:.4f}%")
    print("[LOCO-strict] done")


if __name__ == "__main__":
    main()
