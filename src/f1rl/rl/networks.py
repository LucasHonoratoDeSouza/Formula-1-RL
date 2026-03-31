from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal


def _orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.6))
        self.apply(_orthogonal_init)
        _orthogonal_init(self.mean_head, gain=0.01)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std.clamp(-4.5, 1.0).expand_as(mean)
        return mean, log_std

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean, log_std = self.forward(obs)
        return Normal(mean, log_std.exp())

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = self._log_prob_from_raw(dist, raw_action, action)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs)
        clipped = actions.clamp(-0.999999, 0.999999)
        raw_action = torch.atanh(clipped)
        log_prob = self._log_prob_from_raw(dist, raw_action, clipped)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)

    @staticmethod
    def _log_prob_from_raw(dist: Normal, raw_action: torch.Tensor, squashed_action: torch.Tensor) -> torch.Tensor:
        correction = torch.log(1.0 - squashed_action.pow(2) + 1e-6).sum(dim=-1)
        return dist.log_prob(raw_action).sum(dim=-1) - correction


class CentralizedCritic(nn.Module):
    def __init__(self, obs_dim: int, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(_orthogonal_init)
        _orthogonal_init(self.net[-1], gain=1.0)

    def forward(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, state], dim=-1)).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.actor = SquashedGaussianActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.critic = CentralizedCritic(obs_dim=obs_dim, state_dim=state_dim, hidden_dim=hidden_dim)

    def act(self, obs: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, log_prob, entropy = self.actor.sample(obs)
        values = self.critic(obs, state)
        return actions, log_prob, entropy, values

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_prob, entropy = self.actor.evaluate_actions(obs, actions)
        values = self.critic(obs, state)
        return log_prob, entropy, values

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.deterministic(obs)
