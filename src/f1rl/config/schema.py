from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class SeedConfig:
    seed: int = 7


@dataclass(slots=True)
class DriverConfig:
    name: str
    team: str
    engine_power: float
    aero_efficiency: float
    tire_management: float
    brake_efficiency: float
    aggressiveness: float
    start_compound: str = "medium"


@dataclass(slots=True)
class TrackSegmentConfig:
    name: str
    length_m: float
    curvature: float
    grip: float
    drs_zone: bool = False
    pit_entry: bool = False
    pit_exit: bool = False
    overtaking_factor: float = 0.5


@dataclass(slots=True)
class TrackConfig:
    name: str
    laps: int
    pit_time_loss_s: float
    safety_car_duration_steps: int
    segments: List[TrackSegmentConfig] = field(default_factory=list)


@dataclass(slots=True)
class CarConfig:
    mass_kg: float = 798.0
    fuel_capacity_kg: float = 110.0
    max_speed_mps: float = 97.0
    max_accel_mps2: float = 10.5
    max_brake_mps2: float = 18.0
    drag_coefficient: float = 0.0016
    tire_temp_nominal_c: float = 94.0
    tire_temp_window_c: float = 16.0


@dataclass(slots=True)
class RaceConfig:
    time_step_s: float = 0.5
    max_steps: int = 2200
    nearby_opponents: int = 3
    random_weather_scale: float = 0.025
    initial_wetness: float = 0.08
    crash_damage_threshold: float = 0.92
    drs_gap_m: float = 55.0
    overtake_window_m: float = 14.0
    yellow_flag_steps: int = 12


@dataclass(slots=True)
class PPOConfig:
    rollout_steps: int = 256
    train_iterations: int = 12
    epochs: int = 8
    mini_batch_size: int = 512
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.7
    learning_rate: float = 0.0003
    max_grad_norm: float = 0.8
    hidden_dim: int = 256
    shared_policy: bool = True
    target_kl: float = 0.03
    normalize_advantages: bool = True


@dataclass(slots=True)
class LoggingConfig:
    log_interval: int = 1
    checkpoint_interval: int = 4
    artifact_dir: str = "artifacts"
    run_name: str = "interlagos-mappo-selfplay"


@dataclass(slots=True)
class ProjectConfig:
    seed: SeedConfig
    track: TrackConfig
    car: CarConfig
    race: RaceConfig
    ppo: PPOConfig
    logging: LoggingConfig
    drivers: List[DriverConfig]
