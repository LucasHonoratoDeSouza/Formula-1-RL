import numpy as np

from f1rl.baselines import RuleBasedRaceEngineer
from f1rl.config import load_config
from f1rl.envs import MultiAgentF1Env


def test_environment_step_shapes_and_finiteness() -> None:
    config = load_config("configs/experiments/interlagos_mappo.yaml")
    config.track.laps = 3
    config.race.max_steps = 120
    env = MultiAgentF1Env(config)
    policy = RuleBasedRaceEngineer()

    obs, info = env.reset(seed=123)
    assert obs.shape == (env.num_agents, env.obs_dim)
    assert info["track"] == "Interlagos"

    for _ in range(8):
        actions = policy(obs, env.get_global_state())
        next_obs, rewards, dones, _, step_info = env.step(actions)
        assert next_obs.shape == (env.num_agents, env.obs_dim)
        assert rewards.shape == (env.num_agents,)
        assert dones.shape == (env.num_agents,)
        assert np.isfinite(next_obs).all()
        assert np.isfinite(rewards).all()
        assert "rank" in step_info
        obs = next_obs


def test_environment_can_complete_short_race() -> None:
    config = load_config("configs/experiments/interlagos_mappo.yaml")
    config.track.laps = 2
    config.race.max_steps = 220
    env = MultiAgentF1Env(config)
    policy = RuleBasedRaceEngineer()

    done = False
    obs, _ = env.reset(seed=321)
    while not done:
        obs, _, _, done, info = env.step(policy(obs, env.get_global_state()))

    assert "episode_metrics" in info
    assert len(info["episode_metrics"]["standings"]) == env.num_agents
