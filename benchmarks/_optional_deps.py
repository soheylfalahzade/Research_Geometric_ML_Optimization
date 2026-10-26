"""Stand-ins for heavy optional dependencies.

install_stand_ins() replaces torch, torch_geometric and osmnx with minimal stubs when they
are not installed, so that the torch-free parts of spanner_pipeline (directed repair, stretch
measurement) can be imported on a lightweight machine such as a CI runner. When the real
packages are available nothing is replaced.
"""
import sys
import types
from unittest import mock


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def install_stand_ins():
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
