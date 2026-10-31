"""Torch-free helpers for the strict leave-one-city-out protocol (benchmarks/loco_strict.py)."""
import numpy as np
import pandas as pd

STRICT_COLUMNS = ["seed", "held_out", "trained_on", "n_pairs", "mean_stretch", "median_stretch", "p99_stretch",
                  "max_stretch", "violation_rate_pct", "base_excludes_held_out"]


def filter_training_frame(df, exclude_city=None):
    """Rows of the base-model training table without the held-out city (unchanged when exclude_city is None)."""
    if exclude_city is None:
        return df
    if exclude_city not in set(df["city"]):
        raise ValueError(f"city '{exclude_city}' is not in the training table")
    out = df[df["city"] != exclude_city].reset_index(drop=True)
    if out.empty:
        raise ValueError("no training rows left after excluding the held-out city")
    return out


def stretch_summary(stretches, bound=1.5):
    s = np.asarray(stretches, dtype=float)
    return {"n_pairs": int(s.size), "mean_stretch": float(s.mean()), "median_stretch": float(np.median(s)),
            "p99_stretch": float(np.percentile(s, 99)), "max_stretch": float(s.max()),
            "violation_rate_pct": float((s > bound).mean() * 100)}


def pending_runs(done, cities, seeds):
    """(seed, held_out) pairs not yet in the results table, so an interrupted run can be resumed."""
    finished = set() if done is None or done.empty else set(zip(done["seed"].astype(int), done["held_out"]))
    return [(s, c) for s in seeds for c in cities if (s, c) not in finished]


def summarize_strict(df):
    """One row per held-out city: seeds, mean and standard deviation of the mean stretch, worst max stretch, violation rates."""
    rows = []
    for city, g in df.groupby("held_out", sort=False):
        rows.append({"held_out": city, "n_seeds": int(g["seed"].nunique()), "mean_of_mean": g.mean_stretch.mean(),
                     "std_of_mean": g.mean_stretch.std(), "worst_max": g.max_stretch.max(),
                     "mean_violation_pct": g.violation_rate_pct.mean(), "max_violation_pct": g.violation_rate_pct.max()})
    return pd.DataFrame(rows)
