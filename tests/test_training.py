from pathlib import Path

from f1rl.config import load_config
from f1rl.rl import MAPPOTrainer


def test_mappo_training_smoke(tmp_path: Path) -> None:
    config = load_config("configs/experiments/interlagos_mappo.yaml")
    config.track.laps = 2
    config.race.max_steps = 140
    config.ppo.train_iterations = 1
    config.ppo.rollout_steps = 32
    config.ppo.epochs = 2
    config.ppo.mini_batch_size = 64
    config.logging.artifact_dir = str(tmp_path)
    config.logging.run_name = "pytest-run"

    trainer = MAPPOTrainer(config=config, device="cpu")
    summary = trainer.train()

    assert summary.iterations == 1
    assert (tmp_path / "pytest-run" / "training_summary.json").exists()
    assert list((tmp_path / "pytest-run").glob("checkpoint_iter_*.pt"))
