from __future__ import annotations

import time
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from f1rl.config.schema import ProjectConfig
from f1rl.envs.race import MultiAgentF1Env
from f1rl.rl.buffer import RolloutBatch, RolloutBuffer
from f1rl.rl.networks import ActorCritic
from f1rl.utils.io import dump_json, ensure_dir
from f1rl.utils.seeding import seed_everything


@dataclass(slots=True)
class TrainingSummary:
    run_name: str
    iterations: int
    mean_reward: float
    best_episode_reward: float
    last_episode_standings: list[str]
    elapsed_s: float


class MAPPOTrainer:
    def __init__(self, config: ProjectConfig, device: str | None = None) -> None:
        seed_everything(config.seed.seed)
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.env = MultiAgentF1Env(config)
        self.model = ActorCritic(
            obs_dim=self.env.obs_dim,
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            hidden_dim=config.ppo.hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.ppo.learning_rate)
        self.buffer = RolloutBuffer(
            rollout_steps=config.ppo.rollout_steps,
            num_agents=self.env.num_agents,
            obs_dim=self.env.obs_dim,
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            device=self.device,
        )
        self.artifact_dir = ensure_dir(Path(config.logging.artifact_dir) / config.logging.run_name)
        self.metrics_path = self.artifact_dir / "metrics.jsonl"
        self.best_reward = -float("inf")
        self.latest_summary: dict[str, Any] = {}
        self.reset_counter = 0

    def collect_rollout(self) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        self.buffer.reset()
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        last_episode_metrics: dict[str, Any] = {}
        obs, _ = self.env.reset(seed=self.config.seed.seed + self.reset_counter)
        self.reset_counter += 1
        state = self.env.get_global_state()
        current_episode_reward = np.zeros(self.env.num_agents, dtype=np.float32)
        current_episode_steps = 0

        while self.buffer.ptr < self.config.ppo.rollout_steps:
            active_masks = self.env.active_mask.copy()
            obs_tensor = torch.tensor(obs, device=self.device)
            state_tensor = torch.tensor(np.repeat(state[None, :], self.env.num_agents, axis=0), device=self.device)
            with torch.no_grad():
                actions, log_probs, _, values = self.model.act(obs_tensor, state_tensor)

            action_np = actions.cpu().numpy()
            log_prob_np = log_probs.cpu().numpy()
            value_np = values.cpu().numpy()
            next_obs, rewards, dones, episode_done, info = self.env.step(action_np)
            self.buffer.add(
                observations=obs,
                state=state,
                actions=action_np,
                log_probs=log_prob_np,
                values=value_np,
                rewards=rewards,
                dones=dones,
                active_masks=active_masks,
            )

            current_episode_reward += rewards
            current_episode_steps += 1
            obs = next_obs
            state = self.env.get_global_state()

            if episode_done:
                episode_rewards.append(float(current_episode_reward.mean()))
                episode_lengths.append(current_episode_steps)
                current_episode_reward = np.zeros(self.env.num_agents, dtype=np.float32)
                current_episode_steps = 0
                last_episode_metrics = info.get("episode_metrics", {})
                obs, _ = self.env.reset(seed=self.config.seed.seed + self.reset_counter)
                self.reset_counter += 1
                state = self.env.get_global_state()

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device)
            state_tensor = torch.tensor(np.repeat(state[None, :], self.env.num_agents, axis=0), device=self.device)
            _, _, _, last_values = self.model.act(obs_tensor, state_tensor)

        self.buffer.compute_returns_and_advantages(
            last_values=last_values.cpu().numpy(),
            gamma=self.config.ppo.gamma,
            gae_lambda=self.config.ppo.gae_lambda,
        )
        rollout_stats = {
            "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else float(current_episode_reward.mean()),
            "episode_reward_max": float(np.max(episode_rewards)) if episode_rewards else float(current_episode_reward.max()),
            "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else float(current_episode_steps),
            "episodes_finished": len(episode_rewards),
            "last_episode_metrics": last_episode_metrics,
        }
        return rollout_stats, obs, state

    def update(self) -> dict[str, float]:
        batch = self.buffer.to_torch()
        advantages = batch.advantages
        valid_mask = batch.active_masks > 0.5
        if self.config.ppo.normalize_advantages and valid_mask.any():
            valid_advantages = advantages[valid_mask]
            advantages = advantages.clone()
            advantages[valid_mask] = (valid_advantages - valid_advantages.mean()) / (valid_advantages.std(unbiased=False) + 1e-8)

        indices = torch.arange(batch.observations.shape[0], device=self.device)
        actor_loss_total = 0.0
        critic_loss_total = 0.0
        entropy_total = 0.0
        kl_total = 0.0
        updates = 0

        for _ in range(self.config.ppo.epochs):
            permutation = indices[torch.randperm(indices.numel(), device=self.device)]
            for start in range(0, permutation.numel(), self.config.ppo.mini_batch_size):
                batch_index = permutation[start : start + self.config.ppo.mini_batch_size]
                minibatch = self._slice_batch(batch, batch_index, advantages)
                minibatch_valid = minibatch["active_masks"] > 0.5
                if not minibatch_valid.any():
                    continue

                new_log_probs, entropy, values = self.model.evaluate_actions(
                    minibatch["observations"],
                    minibatch["states"],
                    minibatch["actions"],
                )
                ratio = torch.exp(new_log_probs - minibatch["log_probs"])
                clipped_ratio = ratio.clamp(1.0 - self.config.ppo.clip_ratio, 1.0 + self.config.ppo.clip_ratio)

                surrogate_1 = ratio * minibatch["advantages"]
                surrogate_2 = clipped_ratio * minibatch["advantages"]
                actor_loss = -torch.min(surrogate_1, surrogate_2)[minibatch_valid].mean()

                value_pred_clipped = minibatch["values"] + (values - minibatch["values"]).clamp(
                    -self.config.ppo.clip_ratio,
                    self.config.ppo.clip_ratio,
                )
                value_loss_unclipped = (values - minibatch["returns"]).pow(2)
                value_loss_clipped = (value_pred_clipped - minibatch["returns"]).pow(2)
                critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped)[minibatch_valid].mean()
                entropy_bonus = entropy[minibatch_valid].mean()

                total_loss = (
                    actor_loss
                    + self.config.ppo.value_coef * critic_loss
                    - self.config.ppo.entropy_coef * entropy_bonus
                )
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.ppo.max_grad_norm)
                self.optimizer.step()

                approx_kl = (minibatch["log_probs"] - new_log_probs)[minibatch_valid].mean().abs()
                actor_loss_total += float(actor_loss.item())
                critic_loss_total += float(critic_loss.item())
                entropy_total += float(entropy_bonus.item())
                kl_total += float(approx_kl.item())
                updates += 1
                if approx_kl.item() > self.config.ppo.target_kl:
                    break

        denom = max(updates, 1)
        return {
            "actor_loss": actor_loss_total / denom,
            "critic_loss": critic_loss_total / denom,
            "entropy": entropy_total / denom,
            "approx_kl": kl_total / denom,
        }

    @staticmethod
    def _slice_batch(batch: RolloutBatch, index: torch.Tensor, advantages: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "observations": batch.observations[index],
            "states": batch.states[index],
            "actions": batch.actions[index],
            "log_probs": batch.log_probs[index],
            "returns": batch.returns[index],
            "advantages": advantages[index],
            "values": batch.values[index],
            "active_masks": batch.active_masks[index],
        }

    def save_checkpoint(self, iteration: int) -> Path:
        checkpoint_path = self.artifact_dir / f"checkpoint_iter_{iteration:03d}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "iteration": iteration,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def evaluate(self, episodes: int = 2) -> dict[str, Any]:
        episode_returns = []
        standings: list[list[str]] = []
        incidents = []
        for _ in range(episodes):
            obs, _ = self.env.reset()
            done = False
            episode_return = np.zeros(self.env.num_agents, dtype=np.float32)
            while not done:
                obs_tensor = torch.tensor(obs, device=self.device)
                with torch.no_grad():
                    actions = self.model.act_deterministic(obs_tensor).cpu().numpy()
                obs, step_reward, _, done, _ = self.env.step(actions)
                episode_return += step_reward
            summary = self.env.last_episode_summary
            episode_returns.append(float(episode_return.mean()))
            standings.append(summary.get("standings", []))
            incidents.append(summary.get("incident_count", 0))

        return {
            "reward_mean": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "standings": standings[-1] if standings else [],
            "incident_mean": float(np.mean(incidents)) if incidents else 0.0,
        }

    def train(self) -> TrainingSummary:
        start_time = time.perf_counter()
        for iteration in range(1, self.config.ppo.train_iterations + 1):
            rollout_stats, _, _ = self.collect_rollout()
            optimization_stats = self.update()
            evaluation_stats = self.evaluate(episodes=1)
            metrics = {
                "iteration": iteration,
                **rollout_stats,
                **optimization_stats,
                **evaluation_stats,
            }
            self._append_metrics(metrics)
            if evaluation_stats["reward_mean"] > self.best_reward:
                self.best_reward = evaluation_stats["reward_mean"]
                self.save_checkpoint(iteration)
            elif iteration % self.config.logging.checkpoint_interval == 0:
                self.save_checkpoint(iteration)
            self.latest_summary = metrics

        elapsed = time.perf_counter() - start_time
        summary = TrainingSummary(
            run_name=self.config.logging.run_name,
            iterations=self.config.ppo.train_iterations,
            mean_reward=float(self.latest_summary.get("reward_mean", 0.0)),
            best_episode_reward=float(self.best_reward),
            last_episode_standings=self.latest_summary.get("standings", []),
            elapsed_s=elapsed,
        )
        dump_json(self.artifact_dir / "training_summary.json", asdict(summary))
        return summary

    def _append_metrics(self, metrics: dict[str, Any]) -> None:
        line = metrics.copy()
        ensure_dir(self.metrics_path.parent)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
