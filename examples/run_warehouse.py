"""Allocate three delivery tasks to five robots and run them in a warehouse with walking people.

    python examples/run_warehouse.py            # prints a summary
    python examples/run_warehouse.py --plot     # also writes docs/demo.png
"""
import argparse

import numpy as np

from mrc import Robot, Task, allocate, execute_team, rrt_star, warehouse

ap = argparse.ArgumentParser()
ap.add_argument("--plot", action="store_true")
args = ap.parse_args()

world = warehouse(seed=1)
robots = [Robot(i, np.array(p, float)) for i, p in enumerate([(1, 1), (2, 1), (3, 1), (1, 2), (2, 2)])]
tasks = [
    Task(0, np.array([7.5, 18.0]), np.array([22.0, 2.0]), weight=35),  # heavy: needs a 2-robot team
    Task(1, np.array([12.5, 2.0]), np.array([22.0, 18.0]), weight=12),
    Task(2, np.array([3.0, 10.0]), np.array([17.5, 18.5]), weight=8),
]
teams, deferred = allocate(tasks, robots)
for team in teams:
    print(f"task {team.task.id}: leader R{team.leader.id}, followers {[f'R{r.id}' for r in team.followers]}")
results = [execute_team(world, team, seed=team.task.id) for team in teams]
for r in results:
    print(r)
print("deferred:", [t.id for t in deferred])

if args.plot:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 6))
    for s in world.shelves:
        ax.add_patch(Rectangle((s.xmin, s.ymin), s.xmax - s.xmin, s.ymax - s.ymin, color="0.6"))
    for t in tasks:
        path = rrt_star(world, t.pickup, t.drop, seed=t.id)
        if path:
            xy = np.array(path)
            ax.plot(xy[:, 0], xy[:, 1], "-o", ms=3, label=f"task {t.id} ({t.weight:.0f} kg)")
        ax.plot(*t.pickup, "k^")
        ax.plot(*t.drop, "ks")
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    ax.set_title("RRT* delivery routes in the warehouse (▲ pickup, ■ drop)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/demo.png", dpi=110)
    print("wrote docs/demo.png")
