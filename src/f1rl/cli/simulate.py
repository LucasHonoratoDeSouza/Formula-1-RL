from __future__ import annotations

import argparse

import torch

from f1rl.baselines import RuleBasedRaceEngineer
from f1rl.config import load_config
from f1rl.rl.mappo import MAPPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic race simulation.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/interlagos_mappo.yaml",
        help="Path to the experiment YAML.",
    )
    parser.add_argument(
        "--policy",
        choices=["heuristic", "checkpoint"],
        default="heuristic",
        help="Policy source for the simulation.",
    )
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path when --policy=checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    trainer = MAPPOTrainer(config=config, device=args.device)

    if args.policy == "checkpoint":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --policy=checkpoint")
        payload = torch.load(args.checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(payload["model_state_dict"])

        def policy(obs, _state):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, device=trainer.device)
                return trainer.model.act_deterministic(obs_tensor).cpu().numpy()

    else:
        heuristic = RuleBasedRaceEngineer()

        def policy(obs, state):
            return heuristic(obs, state)

    result = trainer.env.run_deterministic_race(policy)
    print(
        {
            "standings": result.standings,
            "total_reward": result.total_reward,
            "incidents": result.incidents,
        }
    )


if __name__ == "__main__":
    main()
