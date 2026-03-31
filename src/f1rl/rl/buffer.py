from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class RolloutBatch:
    observations: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    active_masks: torch.Tensor


class RolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.reset()

    def reset(self) -> None:
        shape = (self.rollout_steps, self.num_agents)
        self.ptr = 0
        self.observations = np.zeros((*shape, self.obs_dim), dtype=np.float32)
        self.states = np.zeros((self.rollout_steps, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((*shape, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros(shape, dtype=np.float32)
        self.active_masks = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)

    def add(
        self,
        observations: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        active_masks: np.ndarray,
    ) -> None:
        if self.ptr >= self.rollout_steps:
            raise RuntimeError("RolloutBuffer overflow.")

        self.observations[self.ptr] = observations
        self.states[self.ptr] = state
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = values
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.active_masks[self.ptr] = active_masks
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        gae = np.zeros(self.num_agents, dtype=np.float32)
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]

    def to_torch(self) -> RolloutBatch:
        size = self.ptr * self.num_agents
        observations = torch.tensor(self.observations[: self.ptr].reshape(size, self.obs_dim), device=self.device)
        expanded_states = np.repeat(self.states[: self.ptr], self.num_agents, axis=0)
        states = torch.tensor(expanded_states, device=self.device)
        actions = torch.tensor(self.actions[: self.ptr].reshape(size, self.action_dim), device=self.device)
        log_probs = torch.tensor(self.log_probs[: self.ptr].reshape(size), device=self.device)
        returns = torch.tensor(self.returns[: self.ptr].reshape(size), device=self.device)
        advantages = torch.tensor(self.advantages[: self.ptr].reshape(size), device=self.device)
        values = torch.tensor(self.values[: self.ptr].reshape(size), device=self.device)
        active_masks = torch.tensor(self.active_masks[: self.ptr].reshape(size), device=self.device)
        return RolloutBatch(
            observations=observations,
            states=states,
            actions=actions,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
            values=values,
            active_masks=active_masks,
        )
