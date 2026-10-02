"""Centralised task allocation with load-based team formation (leader + followers)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Task:
    id: int
    pickup: np.ndarray
    drop: np.ndarray
    weight: float  # kg


@dataclass
class Robot:
    id: int
    pos: np.ndarray
    capacity: float = 20.0  # kg a single mobile manipulator can carry
    busy: bool = False


@dataclass
class Team:
    task: Task
    leader: Robot
    followers: list[Robot] = field(default_factory=list)

    @property
    def members(self) -> list[Robot]:
        return [self.leader, *self.followers]


def robots_needed(task: Task, capacity: float) -> int:
    return max(1, math.ceil(task.weight / capacity))


def allocate(tasks: list[Task], robots: list[Robot]) -> tuple[list[Team], list[Task]]:
    """Greedy centralised allocation.

    Heaviest tasks are served first. For each task the closest free robot
    becomes leader and the next-closest free robots join as followers until
    the team can lift the load. Tasks that cannot be staffed are deferred.
    """
    teams, deferred = [], []
    free = [r for r in robots if not r.busy]
    for task in sorted(tasks, key=lambda t: -t.weight):
        if not free:
            deferred.append(task)
            continue
        need = robots_needed(task, free[0].capacity)
        if need > len(free):
            deferred.append(task)
            continue
        free.sort(key=lambda r: np.linalg.norm(r.pos - task.pickup))
        members, free = free[:need], free[need:]
        for r in members:
            r.busy = True
        teams.append(Team(task, members[0], members[1:]))
    return teams, deferred


def formation_offsets(n_followers: int, spacing: float = 0.9) -> list[np.ndarray]:
    """Follower offsets around the shared payload, in the leader's frame (x forward)."""
    slots = [(-1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (-2, 0)]
    return [np.array(slots[i % len(slots)], float) * spacing for i in range(n_followers)]
