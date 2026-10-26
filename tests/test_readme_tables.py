"""README.md tables must equal what the raw result files produce."""
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def test_readme_tables_match_raw_data():
    out = subprocess.run([sys.executable, "benchmarks/render_readme_tables.py", "--check"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
