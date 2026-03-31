from __future__ import annotations

import argparse

from f1rl.config import load_config
from f1rl.rl.mappo import MAPPOTrainer
from f1rl.utils.logging import format_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a MAPPO policy on the F1 multi-agent simulator.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/interlagos_mappo.yaml",
        help="Path to the experiment YAML.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cpu or cuda.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    trainer = MAPPOTrainer(config=config, device=args.device)
    summary = trainer.train()
    print(format_metrics(trainer.latest_summary))
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
