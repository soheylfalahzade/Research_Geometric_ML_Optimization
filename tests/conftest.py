"""Test configuration.

The tests cover the torch-free core of the pipeline (directed repair, parallel-edge
handling, stretch measurement) and the README table renderer. Heavy optional dependencies
are replaced by stand-ins when missing (see benchmarks/_optional_deps.py); the neural
network itself is not exercised here.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from _optional_deps import install_stand_ins  # noqa: E402

install_stand_ins()
