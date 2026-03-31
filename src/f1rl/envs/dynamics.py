from __future__ import annotations

import numpy as np

from f1rl.config.schema import CarConfig
from f1rl.envs.car import COMPOUND_GRIP, COMPOUND_WEAR
from f1rl.envs.track import TrackSegment


def tire_grip(compound_index: int, tire_wear: float, tire_temp_c: float, car: CarConfig) -> float:
    compound_multiplier = float(COMPOUND_GRIP[compound_index])
    wear_penalty = 1.0 - 0.45 * np.clip(tire_wear, 0.0, 1.0)
    thermal_offset = abs(tire_temp_c - car.tire_temp_nominal_c) / car.tire_temp_window_c
    thermal_penalty = 1.0 - 0.14 * np.clip(thermal_offset, 0.0, 1.7)
    return float(max(compound_multiplier * wear_penalty * thermal_penalty, 0.45))


def tire_degradation(
    compound_index: int,
    segment: TrackSegment,
    speed_mps: float,
    commitment: float,
    wetness: float,
    tire_management: float,
    dt: float,
) -> float:
    base = 0.00008 * COMPOUND_WEAR[compound_index]
    load = 1.0 + segment.curvature * 1.7 + speed_mps / 115.0
    management = 1.15 - 0.25 * tire_management
    aggression = 1.0 + 0.45 * max(commitment, 0.0)
    weather = 1.0 + 0.35 * wetness
    return float(base * load * management * aggression * weather * dt)


def target_corner_speed(segment: TrackSegment, grip: float, wetness: float, aero_efficiency: float) -> float:
    curvature_penalty = 57.0 * segment.curvature * (1.0 + wetness * 0.75)
    aero_bonus = 11.0 * aero_efficiency * (1.0 - segment.curvature * 0.55)
    return float(max(46.0 + aero_bonus - curvature_penalty + 19.0 * grip * segment.grip, 24.0))


def longitudinal_acceleration(
    speed_mps: float,
    throttle: float,
    brake: float,
    ers_deploy: float,
    fuel_mix: float,
    car: CarConfig,
    driver_power: float,
    fuel_load: float,
    damage: float,
    drs_enabled: bool,
) -> float:
    effective_mass = car.mass_kg + fuel_load
    power_multiplier = 1.0 + 0.16 * ers_deploy + 0.08 * fuel_mix + 0.12 * driver_power
    damage_penalty = max(1.0 - 0.34 * damage, 0.4)
    accel = car.max_accel_mps2 * throttle * power_multiplier * damage_penalty * (car.mass_kg / effective_mass)
    drag = car.drag_coefficient * (0.84 if drs_enabled else 1.0) * speed_mps * speed_mps
    decel = car.max_brake_mps2 * brake
    return float(accel - decel - drag)
