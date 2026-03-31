from __future__ import annotations

import argparse

import torch

from f1rl.config import load_config
from f1rl.rl.mappo import MAPPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a MAPPO checkpoint in the F1 simulator.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/interlagos_mappo.yaml",
        help="Path to the experiment YAML.",
    )
    parser.add_argument("--checkpoint", type=str, required=False, help="Checkpoint path.")
    parser.add_argument("--episodes", type=int, default=2, help="Number of evaluation episodes.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    trainer = MAPPOTrainer(config=config, device=args.device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(payload["model_state_dict"])
    metrics = trainer.evaluate(episodes=args.episodes)
    print(metrics)


if __name__ == "__main__":
    main()
