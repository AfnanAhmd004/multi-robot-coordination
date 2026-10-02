"""2D warehouse world: static rectangular obstacles (shelves) and moving obstacles (people, carts)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains(self, p: np.ndarray, margin: float = 0.0) -> bool:
        return (self.xmin - margin <= p[0] <= self.xmax + margin) and (self.ymin - margin <= p[1] <= self.ymax + margin)


@dataclass
class MovingObstacle:
    pos: np.ndarray
    vel: np.ndarray
    radius: float = 0.5

    def step(self, dt: float, bounds: tuple[float, float]) -> None:
        self.pos = self.pos + self.vel * dt
        for i in range(2):  # bounce off the walls
            if not 0 <= self.pos[i] <= bounds[i]:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 0, bounds[i])


@dataclass
class World:
    width: float
    height: float
    shelves: list[Rect] = field(default_factory=list)
    movers: list[MovingObstacle] = field(default_factory=list)
    robot_radius: float = 0.4

    def free(self, p: np.ndarray, include_movers: bool = True) -> bool:
        if not (0 <= p[0] <= self.width and 0 <= p[1] <= self.height):
            return False
        if any(s.contains(p, self.robot_radius) for s in self.shelves):
            return False
        if include_movers and any(np.linalg.norm(p - m.pos) < m.radius + self.robot_radius for m in self.movers):
            return False
        return True

    def segment_free(self, a: np.ndarray, b: np.ndarray, step: float = 0.1, include_movers: bool = False) -> bool:
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        return all(self.free(a + (b - a) * t, include_movers) for t in np.linspace(0, 1, n))

    def step(self, dt: float) -> None:
        for m in self.movers:
            m.step(dt, (self.width, self.height))


def warehouse(seed: int = 0, n_movers: int = 3) -> World:
    """Three aisles of shelving plus a few people walking through the space."""
    rng = np.random.default_rng(seed)
    shelves = [Rect(x, 4, x + 1.5, 16) for x in (5, 10, 15)]
    movers = [MovingObstacle(pos=rng.uniform([2, 2], [22, 18]), vel=rng.uniform(-0.6, 0.6, 2)) for _ in range(n_movers)]
    return World(24, 20, shelves, movers)
