"""FinOps RL environment.

A small, dependency-free episodic environment in which an agent starts from an
over-provisioned configuration and learns (via PPO) to reach the *cheapest*
tier that still satisfies demand. State is a real-valued vector; the action is a
discrete tier adjustment. This is the environment the FinOps Agent uses to learn
progressively cheaper configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TIERS = ["small", "medium", "large", "xlarge"]
TIER_FACTORS = [1, 2, 4, 8]
TIER_HOURLY_COST = [0.02, 0.05, 0.10, 0.20]
MAX_TIER = len(TIERS) - 1

# Actions: 0 = scale down, 1 = keep, 2 = scale up
ACTION_DOWN = 0
ACTION_KEEP = 1
ACTION_UP = 2


def minimal_tier_for_demand(demand: int) -> int:
    """Oracle: cheapest tier whose factor satisfies ``demand``."""
    for tier, factor in enumerate(TIER_FACTORS):
        if factor >= demand:
            return tier
    return MAX_TIER


def optimal_config(demand: int) -> tuple[int, int]:
    """Cheapest ``(tier, count)`` with ``count * factor(tier) >= demand``."""
    import math

    best: tuple[int, int] = (MAX_TIER, 1)
    best_cost = float("inf")
    for tier in range(MAX_TIER + 1):
        count = max(1, math.ceil(demand / TIER_FACTORS[tier]))
        cost = TIER_HOURLY_COST[tier] * count
        if cost < best_cost:
            best_cost = cost
            best = (tier, count)
    return best


@dataclass
class StepResult:
    state: list[float]
    reward: float
    done: bool
    info: dict[str, Any]


class FinOpsEnv:
    def __init__(self, demand: int = 2, horizon: int = 8, start_tier: int = MAX_TIER) -> None:
        self.demand = max(demand, 1)
        self.horizon = horizon
        self.start_tier = start_tier
        self.tier = start_tier
        self.step_count = 0

    def reset(self) -> list[float]:
        self.tier = self.start_tier
        self.step_count = 0
        return self._state()

    def _state(self) -> list[float]:
        return [
            self.demand / float(TIER_FACTORS[MAX_TIER]),
            self.tier / float(MAX_TIER),
            self.step_count / float(self.horizon),
        ]

    def step(self, action: int) -> StepResult:
        delta = action - 1  # -1, 0, +1
        self.tier = max(0, min(MAX_TIER, self.tier + delta))
        self.step_count += 1

        cost = TIER_HOURLY_COST[self.tier]
        reward = -cost

        done = self.step_count >= self.horizon
        if done:
            if TIER_FACTORS[self.tier] >= self.demand:
                reward += 10.0  # met demand at the end of the episode
            else:
                reward -= 20.0  # failed to meet demand

        return StepResult(
            state=self._state(),
            reward=reward,
            done=done,
            info={"tier": self.tier, "met_demand": TIER_FACTORS[self.tier] >= self.demand},
        )


def rollout(env: FinOpsEnv, policy: Any) -> tuple[list[list[float]], list[int], list[float], list[float]]:
    """Run one episode under ``policy`` and return (states, actions, rewards, log_probs)."""
    states: list[list[float]] = []
    actions: list[int] = []
    rewards: list[float] = []
    log_probs: list[float] = []

    state = env.reset()
    done = False
    while not done:
        action, log_prob = policy.act(state)
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        result = env.step(action)
        rewards.append(result.reward)
        state = result.state
        done = result.done
    return states, actions, rewards, log_probs
