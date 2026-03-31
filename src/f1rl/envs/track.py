from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from f1rl.config.schema import TrackConfig, TrackSegmentConfig


@dataclass(slots=True)
class TrackSegment:
    name: str
    length_m: float
    curvature: float
    grip: float
    drs_zone: bool
    pit_entry: bool
    pit_exit: bool
    overtaking_factor: float


class TrackLayout:
    def __init__(self, config: TrackConfig) -> None:
        self.name = config.name
        self.laps = config.laps
        self.pit_time_loss_s = config.pit_time_loss_s
        self.safety_car_duration_steps = config.safety_car_duration_steps
        self.segments = [self._segment_from_config(segment) for segment in config.segments]
        if not self.segments:
            raise ValueError("Track layout requires at least one segment.")

        self.length_m = float(sum(segment.length_m for segment in self.segments))
        self.segment_bounds = np.cumsum([segment.length_m for segment in self.segments], dtype=np.float64)

    @staticmethod
    def _segment_from_config(config: TrackSegmentConfig) -> TrackSegment:
        return TrackSegment(
            name=config.name,
            length_m=config.length_m,
            curvature=float(np.clip(config.curvature, 0.0, 1.0)),
            grip=float(np.clip(config.grip, 0.4, 1.4)),
            drs_zone=config.drs_zone,
            pit_entry=config.pit_entry,
            pit_exit=config.pit_exit,
            overtaking_factor=float(np.clip(config.overtaking_factor, 0.1, 1.0)),
        )

    def get_segment_index(self, lap_distance_m: float) -> int:
        normalized = lap_distance_m % self.length_m
        index = int(np.searchsorted(self.segment_bounds, normalized, side="right"))
        return min(index, len(self.segments) - 1)

    def get_segment(self, lap_distance_m: float) -> TrackSegment:
        return self.segments[self.get_segment_index(lap_distance_m)]

    def get_progress(self, distance_m: float) -> float:
        return (distance_m % self.length_m) / self.length_m

    def get_remaining_laps(self, distance_m: float) -> float:
        completed_laps = distance_m / self.length_m
        return max(self.laps - completed_laps, 0.0)

    def summarize(self) -> List[str]:
        return [segment.name for segment in self.segments]
