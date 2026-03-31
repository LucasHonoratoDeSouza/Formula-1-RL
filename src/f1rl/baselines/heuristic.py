from __future__ import annotations

import numpy as np


class RuleBasedRaceEngineer:
    """Simple baseline that encodes a conservative race engineer policy."""

    def __call__(self, observations: np.ndarray, _state: np.ndarray | None = None) -> np.ndarray:
        actions = np.zeros((observations.shape[0], 7), dtype=np.float32)
        speed = observations[:, 0]
        tire_wear = observations[:, 3]
        fuel = observations[:, 5]
        ers = observations[:, 6]
        wetness = observations[:, 12]
        curvature = observations[:, 13]
        drs_zone = observations[:, 15]
        drs_available = observations[:, 16]
        remaining_laps = observations[:, 18]
        opponent_gap = observations[:, 27]
        opponent_speed_delta = observations[:, 28]

        throttle = 0.8 - curvature * 0.55 - wetness * 0.3
        brake = np.clip(curvature * 0.75 + wetness * 0.25 - speed * 0.15, 0.0, 1.0)
        commitment = 0.15 + np.clip(-wetness * 0.5 + (0.45 - tire_wear) * 0.4, -0.8, 0.8)
        ers_deploy = np.clip((drs_zone + drs_available) * 0.55 + (opponent_gap > 0).astype(np.float32) * 0.15, 0.0, 1.0)
        ers_deploy *= (ers > 0.25).astype(np.float32)
        fuel_mix = np.clip(0.45 + (remaining_laps < 0.3).astype(np.float32) * 0.25 - wetness * 0.15, 0.0, 1.0)
        racecraft = np.clip(
            0.5 * (opponent_gap > 0).astype(np.float32)
            - 0.35 * (opponent_gap < 0).astype(np.float32)
            + 0.2 * np.sign(opponent_speed_delta),
            -1.0,
            1.0,
        )
        pit_request = ((tire_wear > 0.7) & (remaining_laps > 0.15) & (fuel > 0.15)).astype(np.float32)

        actions[:, 0] = throttle * 2.0 - 1.0
        actions[:, 1] = brake * 2.0 - 1.0
        actions[:, 2] = np.clip(commitment, -1.0, 1.0)
        actions[:, 3] = ers_deploy * 2.0 - 1.0
        actions[:, 4] = fuel_mix * 2.0 - 1.0
        actions[:, 5] = racecraft
        actions[:, 6] = pit_request * 2.0 - 1.0
        return np.clip(actions, -1.0, 1.0)
