"""The strict leave-one-city-out table is generated from its CSV and refuses an inconsistent run."""
import importlib.util
import os
import shutil
import tempfile

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _mod():
    spec = importlib.util.spec_from_file_location("render_readme_tables", os.path.join(ROOT, "benchmarks", "render_readme_tables.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fixture(tmp, flag=True):
    rows = []
    for city, vals in {"Manhattan": [1.0020, 1.0030], "Eindhoven": [1.0050, 1.0060], "Paris": [1.0020, 1.0021], "Rome": [1.0030, 1.0031]}.items():
        for seed, v in enumerate(vals):
            rows.append({"seed": seed, "held_out": city, "trained_on": "x", "n_pairs": 1000, "mean_stretch": v, "median_stretch": 1.0,
                         "p99_stretch": 1.05, "max_stretch": 1.4 + seed * 0.1, "violation_rate_pct": 0.0 if city != "Eindhoven" else 0.01 * (seed + 1),
                         "base_excludes_held_out": flag})
    pd.DataFrame(rows).to_csv(os.path.join(tmp, "leave_one_city_out_strict.csv"), index=False)
    pd.DataFrame({"Held-out City": ["Manhattan", "Eindhoven", "Paris", "Rome"], "Mean Stretch": ["1.0019", "1.0053", "1.0022", "1.0034"]}).to_csv(
        os.path.join(tmp, "leave_one_city_out_results.csv"), index=False)


def test_table_aggregates_by_city_and_names_the_worst_city():
    m, tmp = _mod(), tempfile.mkdtemp()
    try:
        _fixture(tmp)
        m.RAW = tmp
        t = m.table_loco_strict()
        assert "| Eindhoven | 2 | 1.0055 \u00b1 0.0007 | 1.5000 | 0.0150%" in t
        assert m.loco_strict_glance() == "none above 1.5 in 3 of 4 cities in any seed; Eindhoven: 0.0150% on average over 2 seeds"
    finally:
        shutil.rmtree(tmp)


def test_a_run_that_saw_its_held_out_city_is_rejected():
    m, tmp = _mod(), tempfile.mkdtemp()
    try:
        _fixture(tmp, flag=False)
        m.RAW = tmp
        with pytest.raises(AssertionError):
            m.table_loco_strict()
    finally:
        shutil.rmtree(tmp)
