"""FinOps policies: a deterministic greedy baseline and an RL (PPO) policy.

Both expose the same two methods:

* ``act(state)`` — used inside the PPO rollout loop (returns action + log-prob).
* ``select_config(demand, current_tier, current_count)`` — returns the
  ``(tier_index, count)`` to apply to a plan.

The greedy baseline is a naive "downsize one tier, keep the count" heuristic;
the RL policy learns (or, without torch, falls back to) the analytic optimum
that minimises cost while meeting demand.
"""

from __future__ import annotations

import math
from typing import Any

from cloud_orchestra.rl.env import (
    ACTION_DOWN,
    ACTION_KEEP,
    MAX_TIER,
    TIER_FACTORS,
    FinOpsEnv,
    optimal_config,
)


class GreedyPolicy:
    """Baseline: downsize one tier, keep the instance count (naive cost cut)."""

    name = "greedy"

    def act(self, state: list[float]) -> tuple[int, float]:
        current = _decode_tier(state)
        return (ACTION_DOWN if current > 0 else ACTION_KEEP), 0.0

    def select_config(self, demand: int, current_tier: int, current_count: int) -> tuple[int, int]:
        return (max(0, current_tier - 1), max(1, current_count))


class FinOpsPolicy:
    """PPO-trained policy (falls back to the analytic optimum without torch)."""

    name = "ppo"

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._policy: Any = None

    def _ensure(self) -> Any:
        if self._policy is not None:
            return self._policy
        try:
            import torch

            from cloud_orchestra.rl.ppo import MLPPolicy
        except ImportError:
            return None
        policy = MLPPolicy(obs_dim=3, n_actions=3)
        if self._model_path:
            import contextlib

            with contextlib.suppress(FileNotFoundError):
                policy.load_state_dict(torch.load(self._model_path, map_location="cpu"))
        policy.eval()
        self._policy = policy
        return policy

    def act(self, state: list[float]) -> tuple[int, float]:
        policy = self._ensure()
        if policy is None:
            return GreedyPolicy().act(state)
        import torch

        with torch.no_grad():
            logits, _ = policy(torch.tensor([state], dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            action = int(dist.sample().item())
            log_prob = float(dist.log_prob(torch.tensor(action)).item())
        return action, log_prob

    def select_config(self, demand: int, current_tier: int, current_count: int) -> tuple[int, int]:
        policy = self._ensure()
        if policy is None:
            return optimal_config(demand)
        env = FinOpsEnv(demand=demand, start_tier=current_tier, horizon=8)
        state = env.reset()
        done = False
        while not done:
            action, _ = self.act(state)
            result = env.step(action)
            state, done = result.state, result.done
        tier = env.tier
        count = max(1, math.ceil(demand / TIER_FACTORS[tier]))
        return tier, count


def _decode_tier(state: list[float]) -> int:
    return round(state[1] * MAX_TIER)
