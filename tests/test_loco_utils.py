"""Strict leave-one-city-out helpers (benchmarks/loco_utils.py): no torch needed."""
import numpy as np
import pandas as pd
import pytest

from loco_utils import STRICT_COLUMNS, filter_training_frame, pending_runs, stretch_summary, summarize_strict


def _frame():
    return pd.DataFrame({"city": ["A"] * 3 + ["B"] * 2 + ["C"] * 4, "x": range(9)})


def test_excluded_city_never_reaches_training():
    out = filter_training_frame(_frame(), "B")
    assert "B" not in set(out.city) and len(out) == 7 and set(out.city) == {"A", "C"}


def test_none_keeps_every_row_and_unknown_city_is_an_error():
    assert len(filter_training_frame(_frame(), None)) == 9
    with pytest.raises(ValueError):
        filter_training_frame(_frame(), "Z")
    with pytest.raises(ValueError):
        filter_training_frame(_frame()[_frame().city == "A"], "A")


def test_stretch_summary_values_and_bound():
    s = stretch_summary([1.0, 1.0, 1.2, 1.6], bound=1.5)
    assert s["n_pairs"] == 4 and s["max_stretch"] == 1.6 and s["violation_rate_pct"] == 25.0
    assert abs(s["mean_stretch"] - 1.2) < 1e-12 and set(s) <= set(STRICT_COLUMNS)


def test_pending_runs_resumes_without_repeating():
    done = pd.DataFrame({"seed": [0, 0, 1], "held_out": ["A", "B", "A"]})
    assert pending_runs(done, ["A", "B"], range(2)) == [(1, "B")]
    assert len(pending_runs(None, ["A", "B"], range(3))) == 6


def test_summary_has_one_row_per_city():
    df = pd.DataFrame({"seed": [0, 1, 0, 1], "held_out": ["A", "A", "B", "B"], "mean_stretch": [1.0, 1.2, 1.1, 1.1],
                       "max_stretch": [1.4, 1.7, 1.2, 1.2], "violation_rate_pct": [0.0, 0.2, 0.0, 0.0]})
    s = summarize_strict(df).set_index("held_out")
    assert s.loc["A", "n_seeds"] == 2 and abs(s.loc["A", "mean_of_mean"] - 1.1) < 1e-12
    assert s.loc["A", "worst_max"] == 1.7 and s.loc["A", "max_violation_pct"] == 0.2


def test_a_pair_exactly_at_the_bound_is_not_a_violation():
    assert stretch_summary([1.5, 1.5, 1.0], bound=1.5)["violation_rate_pct"] == 0.0
