from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from f1rl.config.schema import ProjectConfig
from f1rl.envs.car import COMPOUND_TO_INDEX, DriverProfile, INDEX_TO_COMPOUND, RaceResult
from f1rl.envs.dynamics import longitudinal_acceleration, target_corner_speed, tire_degradation, tire_grip
from f1rl.envs.track import TrackLayout


@dataclass(slots=True)
class ActionBundle:
    throttle: np.ndarray
    brake: np.ndarray
    commitment: np.ndarray
    ers_deploy: np.ndarray
    fuel_mix: np.ndarray
    racecraft: np.ndarray
    pit_request: np.ndarray


class MultiAgentF1Env:
    action_dim = 7

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.track = TrackLayout(config.track)
        self.num_agents = len(config.drivers)
        self.dt = config.race.time_step_s
        self.max_race_distance = self.track.length_m * self.track.laps
        self.rng = np.random.default_rng(config.seed.seed)
        self.driver_profiles = [
            DriverProfile(
                name=driver.name,
                team=driver.team,
                engine_power=driver.engine_power,
                aero_efficiency=driver.aero_efficiency,
                tire_management=driver.tire_management,
                brake_efficiency=driver.brake_efficiency,
                aggressiveness=driver.aggressiveness,
                start_compound=driver.start_compound,
            )
            for driver in config.drivers
        ]

        self.obs_dim = 27 + config.race.nearby_opponents * 5
        self.state_dim = 6 + self.num_agents * 10
        self.last_episode_summary: dict[str, Any] = {}
        self._drs_available = np.zeros(self.num_agents, dtype=np.float32)
        self.reset()

    @property
    def active_mask(self) -> np.ndarray:
        return (~self.finished & ~self.crashed).astype(np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        spacing = 8.5
        self.step_count = 0
        self.weather_wetness = float(self.config.race.initial_wetness)
        self.yellow_flag_remaining = 0
        self.safety_car_remaining = 0
        self.incident_count = 0
        self.distance = -np.arange(self.num_agents, dtype=np.float32) * spacing
        self.speed = np.full(self.num_agents, 58.0, dtype=np.float32)
        self.tire_wear = np.full(self.num_agents, 0.04, dtype=np.float32)
        self.tire_temp = np.full(
            self.num_agents,
            self.config.car.tire_temp_nominal_c - 4.0,
            dtype=np.float32,
        )
        self.fuel = np.full(
            self.num_agents,
            self.config.car.fuel_capacity_kg * 0.92,
            dtype=np.float32,
        )
        self.ers = np.full(self.num_agents, 0.75, dtype=np.float32)
        self.damage = np.zeros(self.num_agents, dtype=np.float32)
        self.penalty_s = np.zeros(self.num_agents, dtype=np.float32)
        self.finished = np.zeros(self.num_agents, dtype=bool)
        self.crashed = np.zeros(self.num_agents, dtype=bool)
        self.in_pit = np.zeros(self.num_agents, dtype=bool)
        self.pit_time_remaining = np.zeros(self.num_agents, dtype=np.float32)
        self.pit_stops = np.zeros(self.num_agents, dtype=np.int32)
        self.total_reward = np.zeros(self.num_agents, dtype=np.float32)
        self.compound_index = np.array(
            [COMPOUND_TO_INDEX.get(driver.start_compound, 1) for driver in self.driver_profiles],
            dtype=np.int64,
        )
        self.last_rank = self._get_rank_positions()
        self._drs_available = np.zeros(self.num_agents, dtype=np.float32)
        self.last_actions = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)

        return self.get_observation(), self._build_reset_info()

    def _build_reset_info(self) -> dict[str, Any]:
        return {
            "track": self.track.name,
            "teams": [driver.team for driver in self.driver_profiles],
            "drivers": [driver.name for driver in self.driver_profiles],
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }

    def _decode_actions(self, actions: np.ndarray) -> ActionBundle:
        clipped = np.clip(actions, -1.0, 1.0).astype(np.float32)
        return ActionBundle(
            throttle=(clipped[:, 0] + 1.0) * 0.5,
            brake=(clipped[:, 1] + 1.0) * 0.5,
            commitment=clipped[:, 2],
            ers_deploy=(clipped[:, 3] + 1.0) * 0.5,
            fuel_mix=(clipped[:, 4] + 1.0) * 0.5,
            racecraft=clipped[:, 5],
            pit_request=clipped[:, 6] > 0.35,
        )

    def _next_compound(self, agent_idx: int) -> int:
        remaining_laps = self.track.get_remaining_laps(float(self.distance[agent_idx]))
        if remaining_laps > 8.0:
            return COMPOUND_TO_INDEX["hard"]
        if self.tire_wear[agent_idx] > 0.55:
            return COMPOUND_TO_INDEX["medium"]
        return COMPOUND_TO_INDEX["soft"]

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict[str, Any]]:
        if actions.shape != (self.num_agents, self.action_dim):
            raise ValueError(
                f"Expected action tensor with shape {(self.num_agents, self.action_dim)}, got {actions.shape}."
            )

        active_mask = self.active_mask.copy()
        self.last_actions = np.array(actions, copy=True, dtype=np.float32)
        decoded = self._decode_actions(actions)
        old_distance = self.distance.copy()
        old_damage = self.damage.copy()
        old_rank = self._get_rank_positions()
        overtakes = np.zeros(self.num_agents, dtype=np.float32)
        incidents = np.zeros(self.num_agents, dtype=np.float32)
        offtracks = np.zeros(self.num_agents, dtype=np.float32)

        self.step_count += 1
        self.weather_wetness = float(
            np.clip(
                self.weather_wetness + self.rng.normal(0.0, self.config.race.random_weather_scale),
                0.0,
                0.75,
            )
        )

        for idx, driver in enumerate(self.driver_profiles):
            if not active_mask[idx]:
                self.speed[idx] = 0.0
                continue

            if self.in_pit[idx]:
                self.pit_time_remaining[idx] -= self.dt
                self.speed[idx] = 0.0
                if self.pit_time_remaining[idx] <= 0.0:
                    self.in_pit[idx] = False
                    self.compound_index[idx] = self._next_compound(idx)
                    self.tire_wear[idx] = 0.03
                    self.tire_temp[idx] = self.config.car.tire_temp_nominal_c - 6.0
                    self.damage[idx] = max(self.damage[idx] - 0.08, 0.0)
                continue

            segment = self.track.get_segment(float(self.distance[idx]))
            grip = tire_grip(
                int(self.compound_index[idx]),
                float(self.tire_wear[idx]),
                float(self.tire_temp[idx]),
                self.config.car,
            )
            drs_enabled = bool(segment.drs_zone and self._drs_available[idx] and self.safety_car_remaining == 0)
            accel = longitudinal_acceleration(
                speed_mps=float(self.speed[idx]),
                throttle=float(decoded.throttle[idx]),
                brake=float(decoded.brake[idx]),
                ers_deploy=float(decoded.ers_deploy[idx]) * float(self.ers[idx]),
                fuel_mix=float(decoded.fuel_mix[idx]),
                car=self.config.car,
                driver_power=driver.engine_power,
                fuel_load=float(self.fuel[idx]),
                damage=float(self.damage[idx]),
                drs_enabled=drs_enabled,
            )
            corner_speed = target_corner_speed(
                segment=segment,
                grip=grip,
                wetness=float(self.weather_wetness),
                aero_efficiency=driver.aero_efficiency,
            )
            self.speed[idx] = float(np.clip(self.speed[idx] + accel * self.dt, 0.0, self.config.car.max_speed_mps))

            if self.safety_car_remaining > 0:
                self.speed[idx] = min(self.speed[idx], 62.0)

            overspeed = (self.speed[idx] - corner_speed) / max(corner_speed, 1.0)
            if overspeed > 0.02:
                damage_increase = overspeed * (0.05 + max(decoded.commitment[idx], 0.0) * 0.08)
                damage_increase *= 1.0 + 0.25 * driver.aggressiveness + 0.6 * self.weather_wetness
                self.damage[idx] = float(np.clip(self.damage[idx] + damage_increase, 0.0, 1.0))
                self.speed[idx] = min(self.speed[idx], corner_speed * (0.94 - min(overspeed, 0.25) * 0.18))
                offtracks[idx] = 1.0

                crash_risk = overspeed * 0.45 + max(decoded.commitment[idx], 0.0) * 0.12 + self.weather_wetness * 0.18
                if self.rng.random() < crash_risk:
                    self.crashed[idx] = True
                    self.speed[idx] = 0.0
                    incidents[idx] = 1.0
                    self.incident_count += 1
                    self.safety_car_remaining = max(
                        self.safety_car_remaining,
                        self.track.safety_car_duration_steps,
                    )
                    continue

            self.distance[idx] += self.speed[idx] * self.dt
            self.fuel[idx] = float(
                np.clip(
                    self.fuel[idx]
                    - (0.010 + 0.014 * decoded.throttle[idx] + 0.006 * decoded.fuel_mix[idx])
                    * (1.0 + self.speed[idx] / self.config.car.max_speed_mps)
                    * self.dt,
                    0.0,
                    self.config.car.fuel_capacity_kg,
                )
            )
            if self.fuel[idx] <= 0.0:
                self.speed[idx] *= 0.7

            ers_delta = (-0.055 * decoded.ers_deploy[idx] * decoded.throttle[idx] + 0.05 * decoded.brake[idx]) * self.dt
            self.ers[idx] = float(np.clip(self.ers[idx] + ers_delta, 0.0, 1.0))

            self.tire_wear[idx] = float(
                np.clip(
                    self.tire_wear[idx]
                    + tire_degradation(
                        compound_index=int(self.compound_index[idx]),
                        segment=segment,
                        speed_mps=float(self.speed[idx]),
                        commitment=float(decoded.commitment[idx]),
                        wetness=float(self.weather_wetness),
                        tire_management=driver.tire_management,
                        dt=self.dt,
                    ),
                    0.0,
                    1.0,
                )
            )

            thermal_target = (
                self.config.car.tire_temp_nominal_c
                + 20.0 * decoded.throttle[idx]
                + 13.0 * max(decoded.commitment[idx], 0.0) * segment.curvature
                - 18.0 * self.weather_wetness
            )
            self.tire_temp[idx] += (thermal_target - self.tire_temp[idx]) * 0.16 * self.dt

            if decoded.pit_request[idx] and segment.pit_entry:
                self.in_pit[idx] = True
                self.pit_time_remaining[idx] = self.track.pit_time_loss_s
                self.pit_stops[idx] += 1
                self.speed[idx] = 0.0
                continue

            if self.damage[idx] >= self.config.race.crash_damage_threshold or self.tire_wear[idx] >= 0.995:
                self.crashed[idx] = True
                self.speed[idx] = 0.0
                incidents[idx] = 1.0
                self.incident_count += 1
                self.safety_car_remaining = max(self.safety_car_remaining, self.track.safety_car_duration_steps)

        if self.safety_car_remaining > 0:
            self.safety_car_remaining -= 1
        elif self.yellow_flag_remaining > 0:
            self.yellow_flag_remaining -= 1

        overtakes += self._resolve_overtakes(decoded.racecraft)
        new_rank = self._get_rank_positions()
        self._update_drs()

        newly_finished = (~self.finished) & (self.distance >= self.max_race_distance)
        self.finished = self.finished | newly_finished
        self.speed[self.finished] = 0.0

        rewards = self._compute_rewards(
            old_distance=old_distance,
            old_rank=old_rank,
            new_rank=new_rank,
            old_damage=old_damage,
            overtakes=overtakes,
            incidents=incidents,
            offtracks=offtracks,
        )
        rewards *= active_mask
        self.total_reward += rewards
        dones = self.finished | self.crashed
        episode_done = bool(dones.all() or self.step_count >= self.config.race.max_steps)

        info: dict[str, Any] = {
            "rank": (new_rank + 1).tolist(),
            "wetness": self.weather_wetness,
            "safety_car": bool(self.safety_car_remaining > 0),
            "active_mask": active_mask.tolist(),
            "pit_stops": self.pit_stops.tolist(),
            "compounds": [INDEX_TO_COMPOUND[int(index)] for index in self.compound_index],
        }
        if episode_done:
            info["episode_metrics"] = self._build_episode_summary()

        return self.get_observation(), rewards.astype(np.float32), dones.astype(np.float32), episode_done, info

    def _build_episode_summary(self) -> dict[str, Any]:
        order = np.argsort(-self.distance)
        standings = [f"{self.driver_profiles[idx].team} / {self.driver_profiles[idx].name}" for idx in order]
        summary = {
            "track": self.track.name,
            "standings": standings,
            "incident_count": int(self.incident_count),
            "pit_stops": self.pit_stops.astype(int).tolist(),
            "total_reward": self.total_reward.astype(float).round(4).tolist(),
        }
        self.last_episode_summary = summary
        return summary

    def _resolve_overtakes(self, racecraft: np.ndarray) -> np.ndarray:
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        if self.safety_car_remaining > 0:
            return rewards

        order = np.argsort(-self.distance)
        for front_pos in range(self.num_agents - 1):
            front = order[front_pos]
            back = order[front_pos + 1]
            if self.finished[front] or self.finished[back] or self.crashed[front] or self.crashed[back]:
                continue

            gap = self.distance[front] - self.distance[back]
            if gap < 0.0 or gap > self.config.race.overtake_window_m:
                continue

            segment = self.track.get_segment(float(self.distance[back]))
            speed_delta = float(self.speed[back] - self.speed[front])
            attack_score = (
                0.22 * speed_delta / 10.0
                + 0.24 * max(racecraft[back], 0.0)
                - 0.18 * min(racecraft[front], 0.0)
                + 0.10 * (self.ers[back] - self.ers[front])
                + 0.08 * (self.tire_wear[front] - self.tire_wear[back])
                + 0.10 * segment.overtaking_factor
                + (0.08 if segment.drs_zone and self._drs_available[back] else 0.0)
            )
            incident_score = (
                0.03
                + 0.08 * max(racecraft[back], 0.0)
                + 0.06 * max(-racecraft[front], 0.0)
                + 0.05 * self.weather_wetness
            )
            if self.rng.random() < incident_score and abs(speed_delta) > 5.0:
                damage_spike = 0.08 + 0.05 * self.rng.random()
                self.damage[front] = float(np.clip(self.damage[front] + damage_spike, 0.0, 1.0))
                self.damage[back] = float(np.clip(self.damage[back] + damage_spike, 0.0, 1.0))
                self.speed[front] *= 0.82
                self.speed[back] *= 0.8
                self.yellow_flag_remaining = max(self.yellow_flag_remaining, self.config.race.yellow_flag_steps)
                self.incident_count += 1

            if attack_score > self.rng.uniform(-0.1, 0.25):
                advantage = max(1.5, 1.5 + speed_delta * 0.1)
                self.distance[back] = self.distance[front] + advantage
                self.distance[front] -= 0.75
                rewards[back] += 0.12
                rewards[front] -= 0.05

        return rewards

    def _update_drs(self) -> None:
        order = np.argsort(-self.distance)
        self._drs_available[:] = 0.0
        for pos in range(1, self.num_agents):
            current = order[pos]
            ahead = order[pos - 1]
            gap = self.distance[ahead] - self.distance[current]
            if gap <= self.config.race.drs_gap_m:
                self._drs_available[current] = 1.0

    def _get_rank_positions(self) -> np.ndarray:
        order = np.argsort(-self.distance)
        rank = np.empty(self.num_agents, dtype=np.int64)
        rank[order] = np.arange(self.num_agents, dtype=np.int64)
        return rank

    def _compute_rewards(
        self,
        old_distance: np.ndarray,
        old_rank: np.ndarray,
        new_rank: np.ndarray,
        old_damage: np.ndarray,
        overtakes: np.ndarray,
        incidents: np.ndarray,
        offtracks: np.ndarray,
    ) -> np.ndarray:
        progress = (self.distance - old_distance) / self.track.length_m
        rank_delta = old_rank - new_rank
        damage_delta = self.damage - old_damage
        tire_health_bonus = np.clip(0.82 - self.tire_wear, -0.2, 0.25) * 0.04
        ers_window_bonus = (0.55 - np.abs(self.ers - 0.55)) * 0.03
        pit_penalty = self.in_pit.astype(np.float32) * 0.06
        finish_bonus = np.zeros(self.num_agents, dtype=np.float32)

        newly_finished = self.finished & (old_distance < self.max_race_distance)
        if newly_finished.any():
            finish_positions = self._get_rank_positions()
            normalized = 1.0 - finish_positions / max(self.num_agents - 1, 1)
            finish_bonus += newly_finished.astype(np.float32) * (1.5 + 4.5 * normalized)

        return (
            progress * 3.2
            + rank_delta * 0.34
            + overtakes
            + tire_health_bonus
            + ers_window_bonus
            - damage_delta * 3.5
            - incidents * 8.0
            - offtracks * 0.45
            - pit_penalty
            + finish_bonus
        ).astype(np.float32)

    def get_global_state(self) -> np.ndarray:
        rank = self._get_rank_positions()
        state = np.zeros((self.num_agents, 10), dtype=np.float32)
        for idx in range(self.num_agents):
            state[idx] = np.array(
                [
                    self.distance[idx] / self.max_race_distance,
                    self.track.get_progress(float(self.distance[idx])),
                    self.speed[idx] / self.config.car.max_speed_mps,
                    rank[idx] / max(self.num_agents - 1, 1),
                    self.tire_wear[idx],
                    self.fuel[idx] / self.config.car.fuel_capacity_kg,
                    self.ers[idx],
                    self.damage[idx],
                    float(self.in_pit[idx]),
                    float(self.compound_index[idx]) / 2.0,
                ],
                dtype=np.float32,
            )

        extras = np.array(
            [
                self.weather_wetness,
                float(self.safety_car_remaining > 0),
                float(self.yellow_flag_remaining > 0),
                self.step_count / self.config.race.max_steps,
                self.track.laps,
                self.track.length_m / 1000.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([extras, state.reshape(-1)], axis=0).astype(np.float32)

    def get_observation(self) -> np.ndarray:
        rank = self._get_rank_positions()
        observations = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)

        for idx, driver in enumerate(self.driver_profiles):
            segment = self.track.get_segment(float(self.distance[idx]))
            base = [
                self.speed[idx] / self.config.car.max_speed_mps,
                self.track.get_progress(float(self.distance[idx])),
                rank[idx] / max(self.num_agents - 1, 1),
                self.tire_wear[idx],
                (self.tire_temp[idx] - 50.0) / 70.0,
                self.fuel[idx] / self.config.car.fuel_capacity_kg,
                self.ers[idx],
                self.damage[idx],
                min(self.penalty_s[idx] / 10.0, 1.0),
                float(self.in_pit[idx]),
                float(self.finished[idx]),
                float(self.crashed[idx]),
                self.weather_wetness,
                segment.curvature,
                segment.grip,
                float(segment.drs_zone),
                self._drs_available[idx],
                float(self.safety_car_remaining > 0),
                self.track.get_remaining_laps(float(self.distance[idx])) / self.track.laps,
                driver.engine_power,
                driver.aero_efficiency,
                driver.tire_management,
                driver.brake_efficiency,
                driver.aggressiveness,
                float(self.compound_index[idx] == 0),
                float(self.compound_index[idx] == 1),
                float(self.compound_index[idx] == 2),
            ]

            opponent_features: list[float] = []
            other_indices = [candidate for candidate in range(self.num_agents) if candidate != idx]
            other_indices.sort(key=lambda candidate: abs(self.distance[candidate] - self.distance[idx]))
            for candidate in other_indices[: self.config.race.nearby_opponents]:
                opponent_features.extend(
                    [
                        np.clip((self.distance[candidate] - self.distance[idx]) / 120.0, -1.5, 1.5),
                        np.clip((self.speed[candidate] - self.speed[idx]) / 40.0, -1.5, 1.5),
                        rank[candidate] / max(self.num_agents - 1, 1),
                        self.ers[candidate],
                        self.tire_wear[candidate],
                    ]
                )

            while len(opponent_features) < self.config.race.nearby_opponents * 5:
                opponent_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            observations[idx] = np.array(base + opponent_features, dtype=np.float32)

        return observations

    def run_deterministic_race(self, policy_fn) -> RaceResult:
        self.reset()
        done = False
        while not done:
            obs = self.get_observation()
            actions = policy_fn(obs, self.get_global_state())
            _, _, _, done, _ = self.step(actions)

        order = np.argsort(-self.distance)
        standings = [f"{self.driver_profiles[idx].team} / {self.driver_profiles[idx].name}" for idx in order]
        return RaceResult(
            standings=standings,
            total_reward=self.total_reward.astype(float).tolist(),
            incidents=int(self.incident_count),
        )
