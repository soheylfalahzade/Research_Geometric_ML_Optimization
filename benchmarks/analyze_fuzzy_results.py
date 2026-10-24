"""
analyze_fuzzy_results.py
===========================
Summary of all fuzzy_pruning_*.csv files produced so far.
counts only the seeds for which sparsification_pct is equal among E, C, and D
(within a tolerance of 0.001%), exactly equal — i.e. the comparison was fair.

How to run:
    python benchmarks/analyze_fuzzy_results.py
"""

import glob
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_RUNS_DIR = PROJECT_ROOT / "results" / "raw_runs"
TOLERANCE = 0.001


def load_all_runs():
    files = sorted(glob.glob(str(RAW_RUNS_DIR / "fuzzy_pruning_*.csv")))
    print(f"files found ({len(files)} files):")
    for f in files:
        print(f"  - {Path(f).name}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["_source_file"] = Path(f).name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def filter_valid_rows(df):
    valid_keys = set()
    for (city, source, seed), group in df.groupby(["city", "_source_file", "seed"]):
        e_rows = group[group["variant"] == "E_Fuzzy_Pruning"]
        c_rows = group[group["variant"] == "C_GA_Threshold_plus_Feedback"]
        d_rows = group[group["variant"] == "D_Random_Pruning(matched)"]
        if len(e_rows) == 1 and len(c_rows) == 1 and len(d_rows) == 1:
            spars_vals = [e_rows["sparsification_pct"].values[0],
                          c_rows["sparsification_pct"].values[0],
                          d_rows["sparsification_pct"].values[0]]
            if max(spars_vals) - min(spars_vals) < TOLERANCE:
                valid_keys.add((city, source, seed))

    mask = df.apply(lambda row: (row["city"], row["_source_file"], row["seed"]) in valid_keys, axis=1)
    return df[mask].copy()


def main():
    df_all = load_all_runs()
    print(f"\ntotal raw rows: {len(df_all)}")

    df_valid = filter_valid_rows(df_all)
    print(f"valid rows (E/C/D exactly matched in sparsification): {len(df_valid)}\n")

    print("=" * 90)
    print("final summary per city:")
    print("=" * 90)
    summary = df_valid.groupby(["city", "variant"])[["sparsification_pct", "max_stretch", "n_repairs"]].agg(
        ["mean", "std", "count"]
    )
    print(summary.to_string())

    print("\n" + "=" * 90)
    print("direct comparison: is Fuzzy (E) better than C and D?")
    print("=" * 90)
    for city in df_valid["city"].unique():
        sub = df_valid[df_valid["city"] == city]
        e_r = sub[sub["variant"] == "E_Fuzzy_Pruning"]["n_repairs"]
        c_r = sub[sub["variant"] == "C_GA_Threshold_plus_Feedback"]["n_repairs"]
        d_r = sub[sub["variant"] == "D_Random_Pruning(matched)"]["n_repairs"]
        if len(e_r) and len(c_r) and len(d_r):
            e_m, c_m, d_m = e_r.mean(), c_r.mean(), d_r.mean()
            print(f"  {city} (n={len(e_r)}): E={e_m:.1f} | C={c_m:.1f} | D={d_m:.1f}  "
                  f"-> E vs C: {(c_m-e_m)/c_m*100:+.1f}% | E vs D: {(d_m-e_m)/d_m*100:+.1f}%")

    out_path = RAW_RUNS_DIR / "CLEAN_fuzzy_combined_results.csv"
    df_valid.to_csv(out_path, index=False)
    print(f"\n✅ clean data saved: {out_path}")


if __name__ == "__main__":
    main()