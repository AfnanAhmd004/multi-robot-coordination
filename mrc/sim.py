"""Run a team through pickup and delivery, replanning when a moving obstacle blocks the path."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .allocation import Team, formation_offsets
from .perception import SimulatedDetector
from .rrt_star import rrt_star
from .world import World


@dataclass
class RunStats:
    reached: bool = False
    steps: int = 0
    replans: int = 0
    waits: int = 0
    min_clearance: float = np.inf
    trajectory: list[np.ndarray] = field(default_factory=list)


def _blocked(world: World, pos: np.ndarray, waypoint: np.ndarray, lookahead: float) -> bool:
    d = waypoint - pos
    n = np.linalg.norm(d)
    probe = pos + (d / n * min(lookahead, n) if n > 0 else 0)
    return not world.segment_free(pos, probe, include_movers=True)


def drive(world: World, start, goal, speed: float = 1.0, dt: float = 0.2, lookahead: float = 1.5,
          max_steps: int = 2000, seed: int = 0) -> RunStats:
    """Follow an RRT* path; wait or replan when a dynamic obstacle enters the look-ahead window."""
    stats = RunStats()
    pos = np.asarray(start, float)
    goal = np.asarray(goal, float)
    path = rrt_star(world, pos, goal, seed=seed)
    if path is None:
        return stats
    wp = 1
    for t in range(max_steps):
        world.step(dt)
        stats.steps = t + 1
        stats.trajectory.append(pos.copy())
        if world.movers:
            stats.min_clearance = min(stats.min_clearance, min(np.linalg.norm(pos - m.pos) - m.radius - world.robot_radius for m in world.movers))
        if np.linalg.norm(pos - goal) < 0.3:
            stats.reached = True
            return stats
        if world.movers:  # yield: step away from any mover that is about to make contact
            nearest = min(world.movers, key=lambda m: np.linalg.norm(pos - m.pos))
            away = pos - nearest.pos
            gap = np.linalg.norm(away) - nearest.radius - world.robot_radius
            if gap < 0.6 and np.linalg.norm(away) > 0:
                retreat = pos + away / np.linalg.norm(away) * 1.5 * speed * dt
                if world.free(retreat, include_movers=False):
                    pos = retreat
                    stats.waits += 1
                    continue
        if _blocked(world, pos, path[wp], lookahead):
            stats.waits += 1
            if stats.waits % 10 == 0:  # obstacle not clearing: replan around the static map
                new = rrt_star(world, pos, goal, seed=seed + stats.replans + 1)
                if new is not None:
                    path, wp = new, 1
                    stats.replans += 1
            continue
        d = path[wp] - pos
        n = np.linalg.norm(d)
        if n <= speed * dt:
            pos = path[wp].copy()
            wp = min(wp + 1, len(path) - 1)
        else:
            pos = pos + d / n * speed * dt
    return stats


def execute_team(world: World, team: Team, seed: int = 0) -> dict:
    """Leader drives to pickup, the team detects and grasps the payload, then delivers in formation."""
    detector = SimulatedDetector(seed=seed)
    to_pick = drive(world, team.leader.pos, team.task.pickup, seed=seed)
    if not to_pick.reached:
        return {"task": team.task.id, "success": False, "stage": "approach"}
    seen = detector.detect(team.task.pickup, {"payload": team.task.pickup})
    if not seen:
        return {"task": team.task.id, "success": False, "stage": "detection"}
    to_drop = drive(world, team.task.pickup, team.task.drop, speed=0.7, seed=seed + 100)  # slower when loaded
    offsets = formation_offsets(len(team.followers))
    final = to_drop.trajectory[-1] if to_drop.trajectory else team.task.pickup
    for r, off in zip(team.followers, offsets):
        r.pos = final + off
    team.leader.pos = final
    return {
        "task": team.task.id,
        "success": to_drop.reached,
        "team_size": len(team.members),
        "steps": to_pick.steps + to_drop.steps,
        "replans": to_pick.replans + to_drop.replans,
        "min_clearance": min(to_pick.min_clearance, to_drop.min_clearance),
    }
