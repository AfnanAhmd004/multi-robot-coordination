"""Perception interface.

The original system detected target objects with a YOLO model on an
end-effector RGB-D camera. Here detection sits behind a small interface so
the coordination logic can be tested with a simulated detector, and a real
detector can be plugged in without touching the planner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    label: str
    position: np.ndarray
    confidence: float


class Detector(Protocol):
    def detect(self, robot_pos: np.ndarray, objects: dict[str, np.ndarray]) -> list[Detection]: ...


class SimulatedDetector:
    """Range-limited detector with position noise and a confidence that decays with distance."""

    def __init__(self, max_range: float = 4.0, noise: float = 0.05, seed: int = 0):
        self.max_range, self.noise = max_range, noise
        self.rng = np.random.default_rng(seed)

    def detect(self, robot_pos, objects):
        out = []
        for label, pos in objects.items():
            d = np.linalg.norm(pos - robot_pos)
            if d <= self.max_range:
                out.append(Detection(label, pos + self.rng.normal(0, self.noise, 2), float(np.exp(-d / self.max_range))))
        return out
