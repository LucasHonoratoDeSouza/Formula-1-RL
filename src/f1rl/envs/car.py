from __future__ import annotations

from dataclasses import dataclass

import numpy as np


COMPOUND_TO_INDEX = {"soft": 0, "medium": 1, "hard": 2}
INDEX_TO_COMPOUND = {value: key for key, value in COMPOUND_TO_INDEX.items()}
COMPOUND_GRIP = np.array([1.04, 0.99, 0.95], dtype=np.float32)
COMPOUND_WEAR = np.array([1.25, 1.0, 0.82], dtype=np.float32)


@dataclass(slots=True)
class DriverProfile:
    name: str
    team: str
    engine_power: float
    aero_efficiency: float
    tire_management: float
    brake_efficiency: float
    aggressiveness: float
    start_compound: str


@dataclass(slots=True)
class RaceResult:
    standings: list[str]
    total_reward: list[float]
    incidents: int
