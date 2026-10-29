"""Degree safeguard and exact-count random pruning (benchmarks/matched_random.py)."""
import networkx as nx
import numpy as np

from matched_random import MIN_SAFE_DEGREE, apply_degree_safeguard, matched_random_removed


def _graph(n=60, p=0.25, seed=3):
    G = nx.gnp_random_graph(n, p, seed=seed, directed=True)
    edge_list = []
    for u, v in G.edges():
        G[u][v]["length"] = 1.0
        edge_list.append((u, v, G[u][v]))
    return G, edge_list


def _legacy_safeguard(G, edge_list, keep_mask):
    """The loop that used to live inside build_and_repair, kept here as the reference."""
    out_d, in_d = dict(G.out_degree()), dict(G.in_degree())
    removed = np.zeros(len(edge_list), dtype=bool)
    for i, (u, v, _d) in enumerate(edge_list):
        if keep_mask[i] or (in_d[v] <= 2) or (out_d[u] <= 2):
            continue
        removed[i] = True
        out_d[u] -= 1
        in_d[v] -= 1
    return removed


def _final_degrees(G, edge_list, removed):
    H = nx.DiGraph(G)
    H.remove_edges_from([(u, v) for (u, v, _d), r in zip(edge_list, removed) if r])
    return H


def test_safeguard_matches_the_previous_in_loop_implementation():
    G, el = _graph()
    for seed in range(5):
        keep = np.random.default_rng(seed).random(len(el)) < 0.4
        assert (apply_degree_safeguard(G, el, keep) == _legacy_safeguard(G, el, keep)).all()


def test_matched_random_hits_the_target_count_exactly():
    G, el = _graph()
    target = int(apply_degree_safeguard(G, el, np.zeros(len(el), dtype=bool)).sum() * 0.5)
    for seed in range(5):
        assert matched_random_removed(G, el, target, seed).sum() == target


def test_matched_random_respects_the_degree_safeguard():
    G, el = _graph()
    removed = matched_random_removed(G, el, 10 ** 6, seed=1)  # ask for more than is possible
    H = _final_degrees(G, el, removed)
    for (u, v, _d), r in zip(el, removed):
        if r:
            assert H.out_degree(u) >= MIN_SAFE_DEGREE and H.in_degree(v) >= MIN_SAFE_DEGREE


def test_matched_random_is_deterministic_per_seed_and_varies_across_seeds():
    G, el = _graph()
    a, b = matched_random_removed(G, el, 40, 7), matched_random_removed(G, el, 40, 7)
    c = matched_random_removed(G, el, 40, 8)
    assert (a == b).all() and not (a == c).all()
