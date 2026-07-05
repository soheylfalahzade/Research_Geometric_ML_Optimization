"""
analyze_clean_results.py
===========================
قبل از هر تحلیلی، این اسکریپت خودش تشخیص می‌دهد کدام سطرهای CSV از
نسخه‌ی صحیح (v3، با تطبیق دقیق D) آمده‌اند و کدام‌ها از نسخه‌های قدیمی
باگ‌دار — و فقط داده‌ی معتبر را نگه می‌دارد.

معیار اعتبار: برای هر (شهر, seed)، sparsification_pct در ردیف‌های
C_GA_Threshold_plus_Feedback و D_Random_Pruning(matched_to_C) باید
دقیقاً (با تلورانس ۰.۰۰۱٪) برابر باشند. اگر نبودند، آن seed از آن فایل
کنار گذاشته می‌شود.

نحوه‌ی اجرا:
    python benchmarks/analyze_clean_results.py
"""

import glob
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_RUNS_DIR = PROJECT_ROOT / "results" / "raw_runs"

TOLERANCE = 0.001  # درصد اختلاف قابل‌قبول برای تشخیص "برابر"


def load_all_runs():
    files = sorted(glob.glob(str(RAW_RUNS_DIR / "adaptive_ga_feedback_*.csv")))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["_source_file"] = Path(f).name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def filter_valid_rows(df):
    """
    فقط seed هایی را نگه می‌دارد که در همان فایل، C و D دقیقاً یک
    sparsification داشته باشند (یعنی از نسخه‌ی v3 صحیح آمده‌اند).
    """
    valid_keys = set()
    for (city, source, seed), group in df.groupby(["city", "_source_file", "seed"]):
        c_rows = group[group["variant"] == "C_GA_Threshold_plus_Feedback"]
        d_rows = group[group["variant"] == "D_Random_Pruning(matched_to_C)"]
        if len(c_rows) == 1 and len(d_rows) == 1:
            c_spars = c_rows["sparsification_pct"].values[0]
            d_spars = d_rows["sparsification_pct"].values[0]
            if abs(c_spars - d_spars) < TOLERANCE:
                valid_keys.add((city, source, seed))

    mask = df.apply(lambda row: (row["city"], row["_source_file"], row["seed"]) in valid_keys, axis=1)
    return df[mask].copy()


def main():
    df_all = load_all_runs()
    print(f"کل ردیف‌های خام (همه‌ی فایل‌ها): {len(df_all)}")

    df_valid = filter_valid_rows(df_all)
    print(f"ردیف‌های معتبر (بعد از فیلتر C/D matched): {len(df_valid)}")

    n_discarded_seeds = len(df_all.groupby(["city", "_source_file", "seed"])) - \
        len(df_valid.groupby(["city", "_source_file", "seed"]))
    print(f"تعداد (شهر, فایل, seed) کنار گذاشته‌شده به‌خاطر داده‌ی کثیف: {n_discarded_seeds}\n")

    print("=" * 90)
    print("خلاصه‌ی نهایی (فقط داده‌ی معتبر) — این تنها جدولی است که باید در مقاله استفاده شود:")
    print("=" * 90)
    summary = df_valid.groupby(["city", "variant"])[["sparsification_pct", "max_stretch", "n_repairs"]].agg(
        ["mean", "std", "count"]
    )
    print(summary.to_string())

    print("\n" + "=" * 90)
    print("مقایسه‌ی مستقیم C در برابر D (درصد بهبود نسبت به تصادف):")
    print("=" * 90)
    for city in df_valid["city"].unique():
        sub = df_valid[df_valid["city"] == city]
        c_repairs = sub[sub["variant"] == "C_GA_Threshold_plus_Feedback"]["n_repairs"]
        d_repairs = sub[sub["variant"] == "D_Random_Pruning(matched_to_C)"]["n_repairs"]
        if len(c_repairs) and len(d_repairs):
            c_mean, d_mean = c_repairs.mean(), d_repairs.mean()
            improvement = (d_mean - c_mean) / d_mean * 100
            n = len(c_repairs)
            print(f"  {city}: C={c_mean:.1f} (n={n}) vs D={d_mean:.1f} (n={len(d_repairs)}) "
                  f"-> بهبود C نسبت به D: {improvement:+.1f}%")

    out_path = RAW_RUNS_DIR / "CLEAN_combined_results.csv"
    df_valid.to_csv(out_path, index=False)
    print(f"\n✅ داده‌ی تمیز در این فایل ذخیره شد: {out_path}")
    print("از این فایل (نه فایل‌های خام تک‌به‌تک) برای هر جدول مقاله استفاده کن.")


if __name__ == "__main__":
    main()