"""Test configuration.

The tests cover the torch-free core of the pipeline (directed repair, parallel-edge
handling, stretch measurement). When torch, torch_geometric or osmnx are not installed
(for example on the lightweight CI runner) they are replaced by minimal stand-ins so the
pipeline module can still be imported. Tests never touch the neural network itself.
"""
import os
import sys
import types
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


try:
    import torch  # noqa: F401
    import torch_geometric  # noqa: F401
except ImportError:
    class _Module:
        def __init__(self, *a, **k):
            pass
    nn = _stub("torch.nn", Module=_Module)
    torch_stub = mock.MagicMock(name="torch")
    torch_stub.nn = nn
    torch_stub.Tensor = type("Tensor", (), {})   # scipy inspects torch.Tensor when torch is importable
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn.functional"] = mock.MagicMock(name="torch.nn.functional")
    nn.functional = sys.modules["torch.nn.functional"]
    _stub("torch_geometric")
    _stub("torch_geometric.nn", SAGEConv=_Module)

try:
    import osmnx  # noqa: F401
except ImportError:
    _stub("osmnx")
