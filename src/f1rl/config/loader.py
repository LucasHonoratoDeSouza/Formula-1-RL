from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, TypeVar

import yaml

from f1rl.config.schema import (
    CarConfig,
    DriverConfig,
    LoggingConfig,
    PPOConfig,
    ProjectConfig,
    RaceConfig,
    SeedConfig,
    TrackConfig,
    TrackSegmentConfig,
)

T = TypeVar("T")


def _drop_unknown(source: Dict[str, Any], valid_keys: Iterable[str]) -> Dict[str, Any]:
    valid = set(valid_keys)
    return {key: value for key, value in source.items() if key in valid}


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    seed = SeedConfig(**_drop_unknown(raw.get("seed", {}), SeedConfig.__dataclass_fields__))
    car = CarConfig(**_drop_unknown(raw.get("car", {}), CarConfig.__dataclass_fields__))
    race = RaceConfig(**_drop_unknown(raw.get("race", {}), RaceConfig.__dataclass_fields__))
    ppo = PPOConfig(**_drop_unknown(raw.get("ppo", {}), PPOConfig.__dataclass_fields__))
    logging = LoggingConfig(**_drop_unknown(raw.get("logging", {}), LoggingConfig.__dataclass_fields__))

    track_section = raw["track"]
    segments = [
        TrackSegmentConfig(**_drop_unknown(segment, TrackSegmentConfig.__dataclass_fields__))
        for segment in track_section.get("segments", [])
    ]
    track_kwargs = _drop_unknown(track_section, TrackConfig.__dataclass_fields__)
    track_kwargs.pop("segments", None)
    track = TrackConfig(**track_kwargs, segments=segments)

    drivers = [
        DriverConfig(**_drop_unknown(driver, DriverConfig.__dataclass_fields__))
        for driver in raw.get("drivers", [])
    ]

    if not drivers:
        raise ValueError("Configuration must define at least one driver entry.")

    return ProjectConfig(
        seed=seed,
        track=track,
        car=car,
        race=race,
        ppo=ppo,
        logging=logging,
        drivers=drivers,
    )
