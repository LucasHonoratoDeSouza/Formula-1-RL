from f1rl.config import load_config


def test_load_config_contains_f1_teams() -> None:
    config = load_config("configs/experiments/interlagos_mappo.yaml")
    assert config.track.name == "Interlagos"
    assert len(config.drivers) == 10
    assert {driver.team for driver in config.drivers} >= {"Ferrari", "McLaren", "Mercedes"}
    assert config.ppo.hidden_dim == 256
