"""Tests of the pipeline's torch-free core: directed repair, parallel-edge
handling and stretch measurement. Each test states the property it protects."""
import numpy as np
import networkx as nx
import pytest

from spanner_pipeline import (compute_global_stretch_scipy, glv_repair_directed,
                              safe_weighted_adjacency)

T = 1.5


def removed_info(edges):
    return [dict(idx=i, u=u, v=v, length=w) for i, (u, v, w) in enumerate(edges)]


def digraph(edges):
    G = nx.DiGraph()
    for u, v, w in edges:
        G.add_edge(u, v, length=w)
    return G


# ---------------------------------------------------------------- parallel edges
def test_parallel_edges_keep_the_minimum_not_the_sum():
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, length=3.0)
    G.add_edge(0, 1, length=10.0)          # the larger parallel edge comes last on purpose
    G.add_edge(1, 0, length=4.0)
    A = safe_weighted_adjacency(G, nodelist=[0, 1]).toarray()
    assert A[0, 1] == 3.0 and A[1, 0] == 4.0


# ------------------------------------------------------------------------ repair
def test_repair_restores_an_edge_whose_detour_is_too_long():
    sparse = digraph([(0, 2, 1.0), (2, 1, 5.0)])           # detour 6.0 > 1.5 * 1.0
    repaired, repairs = glv_repair_directed(sparse, removed_info([(0, 1, 1.0)]), T)
    assert repairs == [0] and repaired.has_edge(0, 1)


def test_repair_skips_an_edge_with_a_short_enough_directed_path():
    sparse = digraph([(0, 2, 1.0), (2, 1, 0.4)])           # detour 1.4 <= 1.5
    repaired, repairs = glv_repair_directed(sparse, removed_info([(0, 1, 1.0)]), T)
    assert repairs == [] and not repaired.has_edge(0, 1)


def test_repair_is_directed_a_reverse_path_does_not_count():
    sparse = digraph([(1, 0, 1.0)])                         # only the opposite direction exists
    repaired, repairs = glv_repair_directed(sparse, removed_info([(0, 1, 1.0)]), T)
    assert repairs == [0]


@pytest.mark.parametrize("seed", range(10))
def test_after_repair_every_removed_edge_is_within_the_bound(seed):
    rng = np.random.default_rng(seed)
    n = 30
    G = nx.gnp_random_graph(n, 0.18, seed=seed, directed=True)
    for i in range(n):                                       # a ring guarantees strong connectivity
        G.add_edge(i, (i + 1) % n)
    for u, v in G.edges():
        G[u][v]["length"] = float(rng.uniform(1.0, 10.0))
    edges = [(u, v, d["length"]) for u, v, d in G.edges(data=True)]
    drop = rng.random(len(edges)) < 0.35
    removed = [e for e, d in zip(edges, drop) if d]
    sparse = G.copy()
    sparse.remove_edges_from([(u, v) for u, v, _ in removed])
    repaired, repairs = glv_repair_directed(sparse, removed_info(removed), T)
    assert set(repairs) <= set(range(len(removed)))
    for u, v, w in removed:
        d = nx.dijkstra_path_length(repaired, u, v, weight="length") if nx.has_path(repaired, u, v) else float("inf")
        assert d <= T * w + 1e-9, f"edge ({u},{v}) violates the bound after repair"


# ----------------------------------------------------------------------- stretch
def test_identical_graphs_have_stretch_one():
    G = digraph([(0, 1, 1.0), (1, 2, 1.0), (2, 0, 2.0)])
    stretches, _ = compute_global_stretch_scipy(G, G.copy())
    assert np.allclose(stretches, 1.0)


def test_known_detour_gives_the_expected_stretch():
    full = digraph([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.5), (2, 0, 3.0)])
    cut = full.copy()
    cut.remove_edge(0, 2)                                    # 0->2 now costs 2.0 instead of 1.5
    stretches, _ = compute_global_stretch_scipy(full, cut)
    assert stretches.max() == pytest.approx(2.0 / 1.5)
    assert stretches.min() == pytest.approx(1.0)


def test_a_subgraph_never_has_stretch_below_one():
    rng = np.random.default_rng(3)
    n = 25
    G = nx.gnp_random_graph(n, 0.25, seed=3, directed=True)
    for i in range(n):
        G.add_edge(i, (i + 1) % n)
    for u, v in G.edges():
        G[u][v]["length"] = float(rng.uniform(1.0, 5.0))
    H = G.copy()
    for u, v in list(G.edges()):
        if rng.random() < 0.2 and (u + 1) % n != v:
            H.remove_edge(u, v)
    stretches, _ = compute_global_stretch_scipy(G, H)
    assert len(stretches) > 0 and stretches.min() >= 1.0 - 1e-9


def test_directed_asymmetry_is_measured_on_the_reverse_graph():
    ring = digraph([(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])  # one-way ring: d(0,1)=1 but d(1,0)=2
    _, asym = compute_global_stretch_scipy(ring, ring.copy())
    assert len(asym) > 0 and max(asym) > 0.0
