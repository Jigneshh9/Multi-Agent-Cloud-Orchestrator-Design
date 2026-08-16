"""FinOps Agent — RL-powered cost optimization.

Given a (possibly over-provisioned) Terraform plan, the agent determines the
capacity demand implied by the alert, consults the RL policy (or the greedy
baseline when RL is disabled for ablation), and rewrites the plan's compute
tiers *and* instance counts to the cheapest configuration that still meets
demand.
"""

from __future__ import annotations

import math

from cloud_orchestra.agents.base import BaseAgent
from cloud_orchestra.providers.cloud import (
    TIER_FACTOR,
    classify_resource,
    estimate_monthly_cost,
    tier_from_name,
)
from cloud_orchestra.rl.env import TIERS, minimal_tier_for_demand
from cloud_orchestra.rl.policy import GreedyPolicy
from cloud_orchestra.schemas import (
    Alert,
    CloudProvider,
    MemoryEntry,
    TerraformPlan,
    TerraformResource,
)

_TIER_NAMES: dict[CloudProvider, dict[int, str]] = {
    CloudProvider.AWS: {0: "t3.small", 1: "t3.medium", 2: "t3.large", 3: "t3.xlarge"},
    CloudProvider.GCP: {
        0: "n1-standard-1",
        1: "n1-standard-2",
        2: "n1-standard-4",
        3: "n1-standard-8",
    },
    CloudProvider.AZURE: {
        0: "Standard_B1s",
        1: "Standard_B2s",
        2: "Standard_B4ms",
        3: "Standard_B8ms",
    },
}

_INSTANCE_KEYS = ("instance_type", "machine_type", "vm_size")


def capacity_demand(alert: Alert) -> int:
    """Approximate compute-capacity multiplier needed to resolve the alert."""
    if alert.problem_class == "cost_anomaly":
        return 1
    current = alert.current_value
    threshold = alert.threshold
    if current is not None and threshold is not None and threshold > 0:
        return max(1, math.ceil(current / threshold))
    return 2


def current_tier_index(plan: TerraformPlan) -> int:
    for resource in plan.resources:
        family = classify_resource(resource.resource_type)
        if family in ("compute", "autoscaling"):
            name = next(
                (resource.attributes[k] for k in _INSTANCE_KEYS if k in resource.attributes),
                "small",
            )
            tier = tier_from_name(str(name))
            return TIERS.index(tier) if tier in TIERS else 2
        if family == "database":
            name = resource.attributes.get("instance_class", "small")
            tier = tier_from_name(str(name))
            return TIERS.index(tier) if tier in TIERS else 2
    return 2


def current_count(plan: TerraformPlan) -> int:
    for resource in plan.resources:
        family = classify_resource(resource.resource_type)
        if family == "autoscaling":
            return int(
                resource.attributes.get("desired_capacity")
                or resource.attributes.get("min_size")
                or 1
            )
        if family == "compute":
            return int(resource.attributes.get("count") or 1)
    return 1


def rewrite_plan(
    plan: TerraformPlan, demand: int, target_tier: int, target_count: int
) -> TerraformPlan:
    names = _TIER_NAMES.get(plan.provider, _TIER_NAMES[CloudProvider.AWS])
    db_tier = minimal_tier_for_demand(demand)
    new_resources: list[TerraformResource] = []
    for resource in plan.resources:
        attrs = dict(resource.attributes)
        family = classify_resource(resource.resource_type)
        if family == "autoscaling":
            for key in _INSTANCE_KEYS:
                if key in attrs:
                    attrs[key] = names[target_tier]
            attrs["desired_capacity"] = target_count
            attrs["min_size"] = target_count
        elif family == "compute":
            for key in _INSTANCE_KEYS:
                if key in attrs:
                    attrs[key] = names[target_tier]
            attrs["count"] = target_count
        elif family == "database":
            attrs["instance_class"] = names[db_tier]
        new_resources.append(
            TerraformResource(
                resource_type=resource.resource_type,
                name=resource.name,
                provider=resource.provider,
                attributes=attrs,
                depends_on=resource.depends_on,
            )
        )
    updated = plan.model_copy(update={"resources": new_resources})
    updated.estimated_monthly_cost_usd = estimate_monthly_cost(updated)
    return updated


class FinOpsAgent(BaseAgent):
    name = "fin_ops"

    async def optimize(
        self, plan: TerraformPlan, alert: Alert, memory_hits: list[MemoryEntry]
    ) -> TerraformPlan:
        async with self.timed(None, "optimize_cost", input_summary=plan.description):
            demand = capacity_demand(alert)
            current = current_tier_index(plan)
            count = current_count(plan)

            if self.ctx.finops_policy is None:
                policy = GreedyPolicy()
                target, target_count = policy.select_config(demand, current, count)
            else:
                target, target_count = self.ctx.finops_policy.select_config(demand, current, count)

            optimized = rewrite_plan(plan, demand, target, target_count)
            saved = max(0.0, plan.estimated_monthly_cost_usd - optimized.estimated_monthly_cost_usd)
            self.ctx.metrics.incr("fin_ops_cost_saved_usd", saved)
            self.trace(
                None,
                "tier_selected",
                input_summary=f"demand={demand}",
                output_summary=f"tier={TIERS[target]} count={target_count}",
                rationale=f"capacity {TIER_FACTOR[TIERS[target]] * target_count} >= demand {demand}",
            )
            return optimized
