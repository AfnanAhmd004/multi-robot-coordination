"""RRT* path planning with shortcut smoothing."""
from __future__ import annotations

import numpy as np

from .world import World


def rrt_star(world: World, start, goal, max_iter: int = 3000, step: float = 1.0, radius: float = 2.5,
             goal_tol: float = 0.5, goal_bias: float = 0.1, seed: int | None = None) -> list[np.ndarray] | None:
    """Plan a collision-free path among static obstacles. Returns waypoints or None."""
    rng = np.random.default_rng(seed)
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    nodes = [start]
    parent = [-1]
    cost = [0.0]
    best_goal = None

    for _ in range(max_iter):
        sample = goal if rng.random() < goal_bias else rng.uniform([0, 0], [world.width, world.height])
        pts = np.asarray(nodes)
        i_near = int(np.argmin(np.linalg.norm(pts - sample, axis=1)))
        d = sample - nodes[i_near]
        dist = np.linalg.norm(d)
        new = sample if dist <= step else nodes[i_near] + d / dist * step
        if not world.free(new, include_movers=False) or not world.segment_free(nodes[i_near], new):
            continue

        # choose the cheapest parent in the neighbourhood
        near = np.where(np.linalg.norm(pts - new, axis=1) <= radius)[0]
        best_p, best_c = i_near, cost[i_near] + np.linalg.norm(new - nodes[i_near])
        for j in near:
            c = cost[j] + np.linalg.norm(new - nodes[j])
            if c < best_c and world.segment_free(nodes[j], new):
                best_p, best_c = j, c
        nodes.append(new)
        parent.append(best_p)
        cost.append(best_c)
        k = len(nodes) - 1

        # rewire neighbours through the new node if that is cheaper
        for j in near:
            c = best_c + np.linalg.norm(nodes[j] - new)
            if c < cost[j] and world.segment_free(new, nodes[j]):
                parent[j], cost[j] = k, c

        if np.linalg.norm(new - goal) <= goal_tol and (best_goal is None or cost[k] < cost[best_goal]):
            best_goal = k

    if best_goal is None:
        return None
    path, i = [goal], best_goal
    while i != -1:
        path.append(nodes[i])
        i = parent[i]
    return shortcut(world, path[::-1])


def shortcut(world: World, path: list[np.ndarray]) -> list[np.ndarray]:
    """Greedy line-of-sight smoothing."""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not world.segment_free(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def path_length(path: list[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))
