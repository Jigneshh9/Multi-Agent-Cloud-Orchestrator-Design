"""A self-contained Proximal Policy Optimization (PPO) trainer.

This is intentionally a compact, readable implementation — enough to train the
FinOps policy convincingly while remaining simple to cite and audit in a
research paper. The module imports PyTorch at the top; the rest of the codebase
only imports it lazily, so the core runs without torch installed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from cloud_orchestra.rl.env import FinOpsEnv


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(obs)
        return self.actor(h), self.critic(h)

    def act(self, obs: torch.Tensor) -> tuple[int, float, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), value


class PPOTrainer:
    def __init__(
        self,
        *,
        obs_dim: int = 3,
        n_actions: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs: int = 4,
    ) -> None:
        self.policy = MLPPolicy(obs_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.history: list[float] = []

    # -- data collection --------------------------------------------------- #
    def _collect(self, env: FinOpsEnv) -> tuple[dict[str, torch.Tensor], float]:
        states: list[list[float]] = []
        actions: list[int] = []
        rewards: list[float] = []
        log_probs: list[float] = []
        values: list[float] = []
        dones: list[float] = []

        state = env.reset()
        done = False
        while not done:
            obs = torch.tensor([state], dtype=torch.float32)
            action, log_prob, value = self.policy.act(obs)
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(float(value.item()))
            result = env.step(action)
            rewards.append(result.reward)
            dones.append(1.0 if result.done else 0.0)
            state = result.state
            done = result.done

        batch = {
            "states": torch.tensor(states, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "log_probs": torch.tensor(log_probs, dtype=torch.float32),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "values": torch.tensor(values, dtype=torch.float32),
            "dones": torch.tensor(dones, dtype=torch.float32),
        }
        return batch, sum(rewards)

    # -- GAE ---------------------------------------------------------------- #
    def _gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if dones[t] else values[t + 1].item() if t + 1 < len(values) else 0.0
            delta = rewards[t].item() + self.gamma * next_value - values[t].item()
            gae = delta + self.gamma * self.lam * gae * (1.0 - dones[t])
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    # -- update ------------------------------------------------------------- #
    def _update(self, batch: dict[str, torch.Tensor]) -> None:
        advantages, returns = self._gae(batch["rewards"], batch["values"], batch["dones"])
        for _ in range(self.epochs):
            logits, values = self.policy(batch["states"])
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch["actions"])
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - batch["log_probs"]).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (returns - values.squeeze(-1)).pow(2).mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # -- train --------------------------------------------------------------- #
    def train(self, env: FinOpsEnv, iterations: int = 60) -> list[float]:
        for _ in range(iterations):
            batch, episode_reward = self._collect(env)
            self._update(batch)
            self.history.append(episode_reward)
        return self.history

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, map_location="cpu"))
