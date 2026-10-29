"""The ablation table is generated from the final-pipeline files only, and stale numbers are caught."""
import importlib.util
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load():
    spec = importlib.util.spec_from_file_location("render_readme_tables", os.path.join(ROOT, "benchmarks", "render_readme_tables.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ablation_uses_only_final_pipeline_files():
    mod = _load()
    for path, _ in mod.load_ablation().values():
        stamp = os.path.basename(path).split("_")[2]
        assert stamp >= mod.ABLATION_FROM


def test_ablation_covers_all_cities_and_variants():
    mod = _load()
    newest = mod.load_ablation()
    assert sorted(newest) == sorted(mod.CITY_ORDER)
    for _, df in newest.values():
        assert set(df.variant) == {k for k, _ in mod.ABL_LABELS}
        assert df.groupby("variant").size().nunique() == 1


def test_edited_ablation_number_is_detected():
    mod = _load()
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    assert mod.render(text) == text
    first = mod.table_ablation().splitlines()[2].split("|")[-2].strip()
    tampered = text.replace(first, "1.0 \u00b1 0.0", 1)
    assert tampered != text and mod.render(tampered) != tampered
