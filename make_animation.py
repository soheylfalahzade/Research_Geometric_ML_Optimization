"""Animated illustration of the pipeline's mechanics on a small synthetic directed road graph.

Stages: (1) the road graph, (2) pruning, (3) per-edge directed repair, (4) verification.

What is real: the repair step is the repository's own glv_repair_directed, and the final stretch
is measured exhaustively on all ordered pairs. What is illustrative: the graph is synthetic and the
pruning scores are an edge-betweenness heuristic, not the trained network.

Run from the repository root:  python make_animation.py   ->   results/figures/pipeline_animation.gif
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import Delaunay

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks"))
from _optional_deps import install_stand_ins  # noqa: E402

install_stand_ins()
from spanner_pipeline import glv_repair_directed  # noqa: E402

T, SEED, N_POINTS, MAX_LEN, PRUNE_FRACTION, MIN_DEGREE = 1.5, 11, 70, 24.0, 0.45, 2
BLUE, RED, GREEN, GREY, DARK = "#1e40af", "#ef4444", "#16a34a", "#cbd5e1", "#111827"


def build_graph():
    rng = np.random.default_rng(SEED)
    P = rng.uniform(0, 100, (N_POINTS, 2))
    tri = Delaunay(P)
    und = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            u, v = sorted((int(s[a]), int(s[b])))
            if np.linalg.norm(P[u] - P[v]) <= MAX_LEN:
                und.add((u, v))
    G = nx.DiGraph()
    for u, v in sorted(und):
        d = float(np.linalg.norm(P[u] - P[v]))
        if rng.random() < 0.22:                       # one-way street
            a, b = (u, v) if rng.random() < 0.5 else (v, u)
            G.add_edge(a, b, length=d)
        else:
            G.add_edge(u, v, length=d)
            G.add_edge(v, u, length=d)
    core = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(core).copy()
    return G, {n: P[n] for n in G.nodes()}


def prune(G):
    score = nx.edge_betweenness_centrality(G, weight="length")
    order = sorted(G.edges(), key=lambda e: score[e])
    target = int(PRUNE_FRACTION * G.number_of_edges())
    out_deg, in_deg = dict(G.out_degree()), dict(G.in_degree())
    removed = []
    for u, v in order:
        if len(removed) >= target:
            break
        if out_deg[u] <= MIN_DEGREE or in_deg[v] <= MIN_DEGREE:     # live-degree safeguard
            continue
        removed.append((u, v))
        out_deg[u] -= 1
        in_deg[v] -= 1
    return removed


def max_stretch(G_full, H, nodes):
    idx = {n: i for i, n in enumerate(nodes)}

    def mat(G):
        r, c, w = [], [], []
        for u, v, d in G.edges(data=True):
            r.append(idx[u]); c.append(idx[v]); w.append(d["length"])
        return csr_matrix((w, (r, c)), shape=(len(nodes), len(nodes)))
    d0, d1 = shortest_path(mat(G_full), directed=True), shortest_path(mat(H), directed=True)
    mask = (d0 > 0) & np.isfinite(d0)
    ratio = d1[mask] / d0[mask]
    return float(ratio.max()), int((ratio > T + 1e-9).sum()), int(mask.sum())


def main():
    G, pos = build_graph()
    nodes = list(G.nodes())
    removed = prune(G)
    sparse = G.copy()
    sparse.remove_edges_from(removed)
    info = [dict(idx=i, u=u, v=v, length=G[u][v]["length"]) for i, (u, v) in enumerate(removed)]
    repaired, repairs = glv_repair_directed(sparse, info, T)
    restored = {removed[i] for i in repairs}
    confirmed = [e for e in removed if e not in restored]
    worst, n_viol, n_pairs = max_stretch(G, repaired, nodes)
    n_edges, n_final = G.number_of_edges(), repaired.number_of_edges()
    print(f"graph: {len(nodes)} nodes, {n_edges} directed edges | pruned {len(removed)} | restored {len(restored)} | "
          f"final {n_final} ({100 * (1 - n_final / n_edges):.1f}% removed) | max stretch {worst:.3f} | pairs above t: {n_viol}/{n_pairs}")

    one_way = [(u, v) for u, v in G.edges() if not G.has_edge(v, u)]
    by_len = sorted(removed, key=lambda e: G[e[0]][e[1]]["length"])
    group = max(1, len(by_len) // 24)

    frames = [("graph", 0)] * 10
    frames += [("prune", k) for k in np.linspace(len(removed) / 18, len(removed), 18).astype(int)]
    frames += [("prune", len(removed))] * 4
    frames += [("repair", k) for k in range(group, len(by_len) + group, group)]
    frames += [("final", 0)] * 18

    fig = plt.figure(figsize=(9.6, 5.2), dpi=80)
    ax = fig.add_axes([0.01, 0.07, 0.56, 0.84])
    ax2 = fig.add_axes([0.62, 0.12, 0.36, 0.70])

    def seg(edges):
        return [[pos[u], pos[v]] for u, v in edges]

    def draw(i):
        stage, k = frames[i]
        ax.clear()
        ax2.clear()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
        pruned_now = set(removed[:k]) if stage == "prune" else set(removed) if stage in ("repair", "final") else set()
        seen = set(by_len[:k]) if stage == "repair" else (set(by_len) if stage == "final" else set())
        back = set(G.edges()) - pruned_now
        if stage == "final":
            back = set(repaired.edges())
        if stage == "repair":
            back = (set(G.edges()) - set(removed)) | (seen & restored)
        ax.add_collection(LineCollection(seg(back), colors=BLUE if stage in ("repair", "final") else "#64748b", linewidths=1.5, zorder=2))
        if stage == "prune":
            ax.add_collection(LineCollection(seg(pruned_now), colors=RED, linewidths=1.6, zorder=3))
        if stage == "repair":
            ax.add_collection(LineCollection(seg([e for e in by_len if e not in seen]), colors=RED, linewidths=1.6, zorder=3))
            ax.add_collection(LineCollection(seg(seen & restored), colors=GREEN, linewidths=2.4, zorder=4))
            ax.add_collection(LineCollection(seg([e for e in seen if e not in restored]), colors=GREY, linewidths=1.0, linestyles=":", zorder=1))
        if stage == "final":
            ax.add_collection(LineCollection(seg(restored), colors=GREEN, linewidths=2.4, zorder=4))
            ax.add_collection(LineCollection(seg(confirmed), colors=GREY, linewidths=1.0, linestyles=":", zorder=1))
        for u, v in one_way:
            if (u, v) in back or (u, v) in restored and stage in ("repair", "final"):
                a, b = pos[u], pos[v]
                m = (a + b) / 2
                ax.annotate("", xy=m + (b - a) * 0.13, xytext=m - (b - a) * 0.13,
                            arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.0, mutation_scale=9), zorder=5)
        pts = np.array([pos[n] for n in nodes])
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=DARK, zorder=6)
        titles = {"graph": "1  Directed road graph (arrows: one-way streets)",
                  "prune": "2  Pruning: lowest-scored edges removed (red)",
                  "repair": "3  Directed repair: a removed edge is restored (green) when its own detour exceeds t",
                  "final": "4  Verification on all ordered pairs"}
        ax.set_title(titles[stage], fontsize=11, loc="left", color=DARK)

        n_rem = len(pruned_now) if stage != "final" else len(removed)
        n_rest = len(seen & restored) if stage == "repair" else (len(restored) if stage == "final" else 0)
        kept = n_edges - n_rem + n_rest
        ax2.axis("off")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        lines = [("directed edges", f"{n_edges}", DARK), ("pruned", f"{n_rem}", RED),
                 ("restored by repair", f"{n_rest}", GREEN), ("edges kept", f"{kept}", BLUE)]
        for j, (lab, val, col) in enumerate(lines):
            y = 0.93 - 0.115 * j
            ax2.text(0.0, y, lab, fontsize=12, color="#475569", va="center")
            ax2.text(1.0, y, val, fontsize=15, color=col, va="center", ha="right", fontweight="bold")
        ax2.plot([0, 1], [0.44, 0.44], color=GREY, lw=1)
        if stage == "final":
            ok = n_viol == 0
            ax2.text(0.0, 0.34, f"removed from the graph: {100 * (1 - kept / n_edges):.1f}%", fontsize=12, color=DARK)
            ax2.text(0.0, 0.25, f"largest pair stretch: {worst:.3f}   (t = {T})", fontsize=12, color=GREEN if ok else RED)
            ax2.text(0.0, 0.16, f"pairs above t: {n_viol} of {n_pairs}", fontsize=12, color=GREEN if ok else RED)
            ax2.text(0.0, 0.05, "the guarantee is per removed edge;\nmeasured here over all pairs", fontsize=9.5, color="#475569", va="bottom")
        else:
            ax2.text(0.0, 0.30, f"t = {T}", fontsize=12, color="#475569")

    fig.text(0.01, 0.012, "Illustration on a synthetic graph. Scores are an edge-betweenness heuristic, not the trained network; "
             "the repair step is the repository's glv_repair_directed.", fontsize=7.5, color="#64748b")
    out = os.path.join("results", "figures", "pipeline_animation.gif")
    anim = FuncAnimation(fig, draw, frames=len(frames), interval=110)
    anim.save(out, writer=PillowWriter(fps=9))
    print("saved", out, "|", len(frames), "frames |", os.path.getsize(out) // 1024, "KiB")


if __name__ == "__main__":
    main()
