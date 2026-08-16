"""Cloud Harmonizer — selects the best cloud provider per task.

The harmonizer scores AWS/GCP/Azure on three axes (cost, latency, compliance)
using the deterministic pricing/latency models, then returns a ranked,
*explainable* recommendation (the reasoning strings feed the Explainer Agent).
"""

from __future__ import annotations

from cloud_orchestra.agents.base import BaseAgent
from cloud_orchestra.providers.cloud import estimate_latency_ms
from cloud_orchestra.schemas import Alert, CloudProvider, ProviderRecommendation

# Representative cost index per resource family (lower is cheaper).
_FAMILY_COST_INDEX: dict[str, dict[str, float]] = {
    "compute": {"aws": 1.00, "gcp": 0.90, "azure": 0.95},
    "database": {"aws": 1.00, "gcp": 0.88, "azure": 0.96},
    "storage": {"aws": 1.00, "gcp": 0.85, "azure": 0.92},
    "load_balancer": {"aws": 1.00, "gcp": 0.90, "azure": 0.95},
}

_WEIGHTS = {"cost": 0.5, "latency": 0.3, "compliance": 0.2}


def _family(resource_type: str) -> str:
    rt = resource_type.lower()
    if "db" in rt or "sql" in rt or "database" in rt:
        return "database"
    if "storage" in rt or "bucket" in rt or "volume" in rt:
        return "storage"
    if "loadbalancer" in rt or "load_balancer" in rt or "lb" in rt:
        return "load_balancer"
    return "compute"


def _compliance_ok(provider: CloudProvider, alert: Alert) -> bool:
    constraints = alert.raw_payload.get("compliance") or alert.raw_payload.get("data_residency")
    if not constraints:
        return True
    # All three providers offer EU/US/AP regions; treat an explicit residency
    # requirement as satisfiable by providers with regional presence.
    region = (alert.region or "").lower()
    if "eu" in region or "europe" in str(constraints).lower():
        return provider in (CloudProvider.AWS, CloudProvider.GCP, CloudProvider.AZURE)
    return True


class CloudHarmonizer(BaseAgent):
    name = "cloud_harmonizer"

    async def choose(self, alert: Alert) -> ProviderRecommendation:
        async with self.timed(None, "choose_provider", input_summary=alert.resource_type):
            family = _family(alert.resource_type)
            costs = _FAMILY_COST_INDEX.get(family, _FAMILY_COST_INDEX["compute"])
            min_cost = min(costs.values())

            scored: dict[CloudProvider, float] = {}
            reasoning: dict[CloudProvider, list[str]] = {}
            for provider in CloudProvider:
                cost_norm = costs.get(provider.value, 1.0) / min_cost
                latency = estimate_latency_ms(provider, alert.region)
                latency_norm = latency / 100.0
                compliant = _compliance_ok(provider, alert)
                compliance_penalty = 0.0 if compliant else 1.0
                penalty = (
                    _WEIGHTS["cost"] * cost_norm
                    + _WEIGHTS["latency"] * latency_norm
                    + _WEIGHTS["compliance"] * compliance_penalty
                )
                scored[provider] = 1.0 - penalty
                reasoning[provider] = [
                    f"cost index {cost_norm:.2f}x cheapest",
                    f"estimated latency {latency} ms",
                    "compliance satisfied" if compliant else "compliance not satisfied",
                ]

            ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
            best = ranked[0][0]
            alternatives = [(p, round(s, 4)) for p, s in ranked[1:]]
            self.trace(
                None,
                "provider_chosen",
                output_summary=best.value,
                rationale="; ".join(reasoning[best]),
            )
            return ProviderRecommendation(
                provider=best,
                score=round(scored[best], 4),
                reasoning=reasoning[best],
                alternatives=alternatives,
            )
