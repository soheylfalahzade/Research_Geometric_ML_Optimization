"""Degree safeguard and exact-count random pruning, free of torch / OSMnx so that it can be tested.

The repair stage keeps an edge whenever removing it would leave an endpoint with a live degree of
MIN_SAFE_DEGREE or less. `apply_degree_safeguard` applies that rule to a proposed keep-mask, and
`matched_random_removed` draws a random removal set that is subject to the same rule and has exactly
the number of removed edges that another variant ended up with, so the repair counts of the variants
can be compared at the same number of pruned edges.
"""
import numpy as np

MIN_SAFE_DEGREE = 2


def _is_bottleneck(u, v, live_out, live_in, min_safe_degree):
    return live_in[v] <= min_safe_degree or live_out[u] <= min_safe_degree


def apply_degree_safeguard(G, edge_list, keep_mask, min_safe_degree=MIN_SAFE_DEGREE):
    """Edges are visited in index order; a proposed removal is refused at a bottleneck. Returns a boolean removed-mask."""
    live_out, live_in = dict(G.out_degree()), dict(G.in_degree())
    removed = np.zeros(len(edge_list), dtype=bool)
    for i, (u, v, _d) in enumerate(edge_list):
        if keep_mask[i] or _is_bottleneck(u, v, live_out, live_in, min_safe_degree):
            continue
        removed[i] = True
        live_out[u] -= 1
        live_in[v] -= 1
    return removed


def matched_random_removed(G, edge_list, n_target, seed, min_safe_degree=MIN_SAFE_DEGREE):
    """Random removal set with exactly n_target removed edges (fewer only if the safeguard makes that impossible)."""
    live_out, live_in = dict(G.out_degree()), dict(G.in_degree())
    removed = np.zeros(len(edge_list), dtype=bool)
    count = 0
    for i in np.random.default_rng(seed).permutation(len(edge_list)):
        if count >= n_target:
            break
        u, v, _d = edge_list[i]
        if _is_bottleneck(u, v, live_out, live_in, min_safe_degree):
            continue
        removed[i] = True
        live_out[u] -= 1
        live_in[v] -= 1
        count += 1
    return removed
