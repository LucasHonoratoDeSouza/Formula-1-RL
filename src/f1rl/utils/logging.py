from __future__ import annotations

from typing import Any


def format_metrics(metrics: dict[str, Any]) -> str:
    ordered = []
    for key in [
        "iteration",
        "episode_reward_mean",
        "episode_length_mean",
        "actor_loss",
        "critic_loss",
        "entropy",
        "approx_kl",
        "reward_mean",
        "incident_mean",
    ]:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                ordered.append(f"{key}={value:.4f}")
            else:
                ordered.append(f"{key}={value}")
    return " | ".join(ordered)
