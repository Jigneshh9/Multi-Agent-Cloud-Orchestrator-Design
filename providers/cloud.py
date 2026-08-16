"""Cloud provider adapters, pricing/latency models and metric simulation.

This module contains the pure cost/latency models consumed by the Cloud
Harmonizer and FinOps agents, plus cloud clients used by the Verifier agent to
observe post-remediation metrics. A deterministic :class:`MockCloudClient`
simulates the effect of a remediation so the closed-loop can be tested offline.
"""

from __future__ import annotations

from typing import Any, Protocol

from cloud_orchestra.core.errors import ProviderError
from cloud_orchestra.schemas import Alert, CloudProvider, TerraformPlan

HOURS_PER_MONTH = 730.0
STORAGE_GB_MONTHLY = 0.02

INSTANCE_TIER_HOURLY = {
    "small": 0.02,
    "medium": 0.05,
    "large": 0.10,
    "xlarge": 0.20,
}
DB_TIER_HOURLY = {
    "small": 0.04,
    "medium": 0.08,
    "large": 0.16,
    "xlarge": 0.32,
}
TIER_FACTOR = {"small": 1, "medium": 2, "large": 4, "xlarge": 8}
PROVIDER_COST_MULTIPLIER = {"aws": 1.0, "gcp": 0.9, "azure": 0.95}

_REGION_LATENCY: dict[str, dict[str, int]] = {
    "us-east-1": {"aws": 8, "gcp": 12, "azure": 14},
    "us-west-2": {"aws": 26, "gcp": 30, "azure": 32},
    "eu-west-1": {"aws": 28, "gcp": 24, "azure": 30},
    "eu-central-1": {"aws": 34, "gcp": 26, "azure": 28},
    "ap-south-1": {"aws": 90, "gcp": 88, "azure": 96},
    "ap-southeast-1": {"aws": 82, "gcp": 78, "azure": 88},
    "ap-northeast-1": {"aws": 70, "gcp": 66, "azure": 74},
}


def tier_from_name(name: str) -> str:
    import re

    n = name.lower()
    m = re.search(r"standard-(\d+)", n)
    if m:
        cores = int(m.group(1))
        if cores <= 1:
            return "small"
        if cores <= 2:
            return "medium"
        if cores <= 4:
            return "large"
        return "xlarge"
    if "2xlarge" in n or "3xlarge" in n or "4xlarge" in n or "xlarge" in n:
        return "xlarge"
    if "large" in n:
        return "large"
    if "medium" in n:
        return "medium"
    return "small"


def _instance_attr(resource: Any, *names: str) -> Any:
    for name in names:
        if name in resource.attributes:
            return resource.attributes[name]
    return None


def classify_resource(resource_type: str) -> str:
    """Return a normalised resource family for a Terraform resource type.

    Order matters: several provider resource names embed substrings of other
    families (e.g. ``google_compute_disk`` contains "compute" but is storage),
    so the most specific families are matched first.
    """
    rt = resource_type.lower()
    if any(k in rt for k in ("autoscaling", "instance_group_manager", "scale_set")):
        return "autoscaling"
    if any(k in rt for k in ("db", "database", "sql")):
        return "database"
    if any(k in rt for k in ("storage", "bucket", "volume", "disk")):
        return "storage"
    if any(k in rt for k in ("loadbalancer", "load_balancer", "forwarding_rule", "elb", "_lb")):
        return "load_balancer"
    if any(k in rt for k in ("instance", "virtual_machine", "compute")):
        return "compute"
    return "other"


def capacity_scale(plan: TerraformPlan) -> int:
    """Total compute capacity multiplier (count x tier factor) of a plan."""
    total = 0
    for resource in plan.resources:
        family = classify_resource(resource.resource_type)
        if family == "compute":
            name = _instance_attr(resource, "instance_type", "machine_type", "vm_size") or "small"
            count = int(_instance_attr(resource, "count") or 1)
            total += count * TIER_FACTOR[tier_from_name(str(name))]
        elif family == "autoscaling":
            name = _instance_attr(resource, "instance_type", "machine_type", "vm_size") or "small"
            count = int(
                _instance_attr(resource, "desired_capacity")
                or _instance_attr(resource, "min_size")
                or 2
            )
            total += count * TIER_FACTOR[tier_from_name(str(name))]
        elif family == "database":
            name = _instance_attr(resource, "instance_class", "tier") or "small"
            total += TIER_FACTOR[tier_from_name(str(name))]
    return max(total, 1)


def storage_added_gb(plan: TerraformPlan) -> float:
    total = 0.0
    for resource in plan.resources:
        if classify_resource(resource.resource_type) == "storage":
            total += float(_instance_attr(resource, "size_gb", "storage_gb") or 0)
    return total


def estimate_monthly_cost(plan: TerraformPlan) -> float:
    mult = PROVIDER_COST_MULTIPLIER.get(plan.provider.value, 1.0)
    total = 0.0
    for resource in plan.resources:
        family = classify_resource(resource.resource_type)
        if family == "compute":
            name = _instance_attr(resource, "instance_type", "machine_type", "vm_size") or "small"
            count = int(_instance_attr(resource, "count") or 1)
            total += INSTANCE_TIER_HOURLY[tier_from_name(str(name))] * HOURS_PER_MONTH * count * mult
        elif family == "autoscaling":
            name = _instance_attr(resource, "instance_type", "machine_type", "vm_size") or "small"
            count = int(
                _instance_attr(resource, "desired_capacity")
                or _instance_attr(resource, "min_size")
                or 2
            )
            total += INSTANCE_TIER_HOURLY[tier_from_name(str(name))] * HOURS_PER_MONTH * count * mult
        elif family == "database":
            name = _instance_attr(resource, "instance_class", "tier") or "small"
            total += DB_TIER_HOURLY[tier_from_name(str(name))] * HOURS_PER_MONTH * mult
        elif family == "storage":
            gb = float(_instance_attr(resource, "size_gb", "storage_gb") or 10)
            total += gb * STORAGE_GB_MONTHLY * mult
    return round(total, 2)


def estimate_latency_ms(provider: CloudProvider, region: str | None) -> int:
    region = region or "us-east-1"
    table = _REGION_LATENCY.get(region, _REGION_LATENCY["us-east-1"])
    return table.get(provider.value, 12)


class CloudClient(Protocol):
    async def query_metric(self, resource_id: str, metric_name: str) -> float | None: ...

    async def apply_effect(self, alert: Alert, plan: TerraformPlan) -> float: ...


class MockCloudClient:
    """Deterministic cloud client that simulates remediation effects."""

    def __init__(self) -> None:
        self._metrics: dict[str, float] = {}

    def set_metric(self, resource_id: str, metric_name: str, value: float) -> None:
        self._metrics[f"{resource_id}:{metric_name}"] = value

    async def query_metric(self, resource_id: str, metric_name: str) -> float | None:
        return self._metrics.get(f"{resource_id}:{metric_name}")

    async def apply_effect(self, alert: Alert, plan: TerraformPlan) -> float:
        """Return the post-remediation metric value for the alert's metric."""
        current = alert.current_value if alert.current_value is not None else 100.0
        threshold = alert.threshold if alert.threshold is not None else 80.0
        pc = alert.problem_class
        scale = capacity_scale(plan)

        if pc in ("high_cpu", "high_memory", "high_latency"):
            after = current / scale
        elif pc == "storage_full":
            added = storage_added_gb(plan)
            after = threshold - 10.0 if added > 0 else current
        elif pc == "db_capacity":
            after = current / scale
        elif pc == "cost_anomaly":
            after = estimate_monthly_cost(plan)
        else:
            after = current / scale

        if pc != "cost_anomaly":
            after = min(after, current) if scale <= 1 else after
        return round(after, 2)


class _SdkCloudClient:
    """Base for real SDK-backed clients (thin, guarded)."""

    provider: CloudProvider

    async def query_metric(self, resource_id: str, metric_name: str) -> float | None:
        raise ProviderError(
            f"real {self.provider.value} metric query requires cloud SDK credentials",
            code=ProviderError.code,
        )

    async def apply_effect(self, alert: Alert, plan: TerraformPlan) -> float:
        # In production this polls CloudWatch/Monitoring until stabilisation.
        raise ProviderError(f"{self.provider.value} apply_effect requires a live deployment")


class AwsCloudClient(_SdkCloudClient):
    provider = CloudProvider.AWS


class GcpCloudClient(_SdkCloudClient):
    provider = CloudProvider.GCP


class AzureCloudClient(_SdkCloudClient):
    provider = CloudProvider.AZURE


def build_cloud_client(provider: CloudProvider | None = None) -> CloudClient:
    # The deterministic mock is the default; swap in SDK clients in production
    # when credentials are configured.
    return MockCloudClient()
