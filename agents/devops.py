"""DevOps Agent — generates Terraform remediation plans.

Two plan generators are provided:

* :class:`RuleBasedPlanGenerator` — deterministic, used for tests, evaluation
  and as the no-LLM ablation baseline.
* :class:`LLMPlanGenerator` — prompts DeepSeek V4 / GPT-4o to emit a typed
  :class:`TerraformPlan` (validated by Pydantic), conditioned on RAG memory.

The agent also exposes :meth:`harden`, which deterministically remediates the
security findings produced by the Review and Red-Team agents.
"""

from __future__ import annotations

from typing import Any

from cloud_orchestra.agents.base import BaseAgent
from cloud_orchestra.core.llm import LLMClient
from cloud_orchestra.providers.cloud import estimate_monthly_cost
from cloud_orchestra.schemas import (
    Alert,
    CloudProvider,
    MemoryEntry,
    SecurityFinding,
    TerraformPlan,
    TerraformResource,
)

# Provider-specific resource names and attribute keys.
_PROVIDER = {
    CloudProvider.AWS: {
        "instance": "aws_instance",
        "autoscale": "aws_autoscaling_group",
        "db": "aws_db_instance",
        "storage": "aws_ebs_volume",
        "lb": "aws_lb",
        "instance_type_attr": "instance_type",
        "db_class_attr": "instance_class",
        "storage_size_attr": "size_gb",
    },
    CloudProvider.GCP: {
        "instance": "google_compute_instance",
        "autoscale": "google_compute_instance_group_manager",
        "db": "google_sql_database_instance",
        "storage": "google_compute_disk",
        "lb": "google_compute_forwarding_rule",
        "instance_type_attr": "machine_type",
        "db_class_attr": "tier",
        "storage_size_attr": "size_gb",
    },
    CloudProvider.AZURE: {
        "instance": "azurerm_linux_virtual_machine",
        "autoscale": "azurerm_virtual_machine_scale_set",
        "db": "azurerm_mssql_database",
        "storage": "azurerm_managed_disk",
        "lb": "azurerm_lb",
        "instance_type_attr": "vm_size",
        "db_class_attr": "sku_name",
        "storage_size_attr": "storage_gb",
    },
}


def _cfg(provider: CloudProvider) -> dict[str, str]:
    return _PROVIDER[provider]


class RuleBasedPlanGenerator:
    """Deterministic alert -> Terraform IR mapping."""

    def generate(
        self, alert: Alert, provider: CloudProvider, memory_hits: list[MemoryEntry]
    ) -> TerraformPlan:
        cfg = _cfg(provider)
        problem = alert.problem_class
        resources: list[TerraformResource] = []

        if problem == "high_cpu":
            resources.append(
                TerraformResource(
                    resource_type=cfg["autoscale"],
                    name="web_asg",
                    provider=provider,
                    attributes={
                        "desired_capacity": 3,
                        "min_size": 2,
                        "max_size": 6,
                        cfg["instance_type_attr"]: "medium",
                        "associate_public_ip_address": True,  # seeded vuln
                        "patch_management": False,  # seeded runtime vuln
                        "ingress": [
                            {"from_port": 80, "to_port": 80, "cidr_blocks": ["0.0.0.0/0"]}
                        ],
                    },
                )
            )
        elif problem == "high_memory":
            resources.append(
                TerraformResource(
                    resource_type=cfg["instance"],
                    name="mem_worker",
                    provider=provider,
                    attributes={cfg["instance_type_attr"]: "large", "count": 2},
                )
            )
        elif problem == "storage_full":
            resources.append(
                TerraformResource(
                    resource_type=cfg["storage"],
                    name="data_volume",
                    provider=provider,
                    attributes={cfg["storage_size_attr"]: 100, "encrypted": False},  # seeded vuln
                )
            )
        elif problem == "db_capacity":
            resources.append(
                TerraformResource(
                    resource_type=cfg["db"],
                    name="app_db",
                    provider=provider,
                    attributes={
                        cfg["db_class_attr"]: "large",
                        "publicly_accessible": True,  # seeded vuln
                        "storage_encrypted": False,
                        "password_rotation_enabled": False,  # seeded runtime vuln
                    },
                )
            )
        elif problem == "high_latency":
            resources.append(
                TerraformResource(
                    resource_type=cfg["lb"],
                    name="front_lb",
                    provider=provider,
                    attributes={"scheme": "internet-facing"},
                )
            )
            resources.append(
                TerraformResource(
                    resource_type=cfg["autoscale"],
                    name="web_asg",
                    provider=provider,
                    attributes={
                        "desired_capacity": 4,
                        "min_size": 2,
                        "max_size": 8,
                        cfg["instance_type_attr"]: "medium",
                    },
                    depends_on=["front_lb"],
                )
            )
        elif problem == "cost_anomaly":
            resources.append(
                TerraformResource(
                    resource_type=cfg["instance"],
                    name="rightsized",
                    provider=provider,
                    attributes={cfg["instance_type_attr"]: "small", "count": 1},
                )
            )
        else:
            resources.append(
                TerraformResource(
                    resource_type=cfg["instance"],
                    name="generic",
                    provider=provider,
                    attributes={cfg["instance_type_attr"]: "medium"},
                )
            )

        plan = TerraformPlan(
            provider=provider,
            provider_config={"region": alert.region or "us-east-1"},
            resources=resources,
            description=f"Remediation for {alert.name} ({problem})",
            tags={"managed-by": "cloud-orchestra"},
        )
        plan.estimated_monthly_cost_usd = estimate_monthly_cost(plan)
        return self._apply_memory(plan, alert, provider, memory_hits)

    @staticmethod
    def _apply_memory(
        plan: TerraformPlan,
        alert: Alert,
        provider: CloudProvider,
        memory_hits: list[MemoryEntry],
    ) -> TerraformPlan:
        """Reuse a proven past deployment for the same problem class (RAG).

        If memory contains a *resolved* deployment for the same problem class
        that is cheaper than the naive plan, adopt its compute configuration.
        This is the mechanism by which persistent memory makes the DevOps agent
        progressively smarter.
        """
        for memory in memory_hits:
            if (
                memory.problem_class == alert.problem_class
                and memory.resolved
                and memory.terraform_plan is not None
            ):
                candidate = memory.terraform_plan
                if candidate.estimated_monthly_cost_usd < plan.estimated_monthly_cost_usd:
                    candidate.provider = provider
                    candidate.estimated_monthly_cost_usd = estimate_monthly_cost(candidate)
                    return candidate
        return plan


class LLMPlanGenerator:
    """LLM-backed generator that emits a validated Terraform IR."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def generate(
        self, alert: Alert, provider: CloudProvider, memory_hits: list[MemoryEntry]
    ) -> TerraformPlan:
        context = "\n".join(
            f"- [{m.problem_class}/{m.provider.value}] {m.summary} (success={m.success})"
            for m in memory_hits[:3]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior DevOps engineer. Produce a Terraform plan as JSON "
                    "matching the TerraformPlan schema (provider, resources[], attributes)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Alert: {alert.name} ({alert.problem_class}) on {alert.resource_id}. "
                    f"Target provider: {provider.value}. "
                    f"Past relevant deployments:\n{context or '(none)'}\n"
                    "Return only JSON."
                ),
            },
        ]
        return await self._llm.complete_structured(messages, TerraformPlan)


class DevOpsAgent(BaseAgent):
    name = "devops"

    def __init__(self, ctx: Any, generator: Any | None = None) -> None:
        super().__init__(ctx)
        self._generator = generator or RuleBasedPlanGenerator()

    async def generate(
        self, alert: Alert, provider: CloudProvider, memory_hits: list[MemoryEntry]
    ) -> TerraformPlan:
        async with self.timed(None, "generate_plan", input_summary=alert.problem_class):
            plan = await self._generate(alert, provider, memory_hits)
            plan.estimated_monthly_cost_usd = estimate_monthly_cost(plan)
        return plan

    async def _generate(
        self, alert: Alert, provider: CloudProvider, memory_hits: list[MemoryEntry]
    ) -> TerraformPlan:
        if isinstance(self._generator, LLMPlanGenerator):
            return await self._generator.generate(alert, provider, memory_hits)
        return self._generator.generate(alert, provider, memory_hits)

    def harden(self, plan: TerraformPlan, findings: list[SecurityFinding]) -> TerraformPlan:
        """Deterministically remediate the security findings in a plan copy."""
        types = {f.vulnerability_type for f in findings}
        new_resources: list[TerraformResource] = []
        for resource in plan.resources:
            attrs = dict(resource.attributes)
            rt = resource.resource_type.lower()
            if "publicly_accessible_database" in types and "db" in rt:
                attrs["publicly_accessible"] = False
            if "open_ingress" in types and "ingress" in attrs:
                attrs["ingress"] = [
                    {"from_port": 80, "to_port": 80, "cidr_blocks": ["10.0.0.0/8"]}
                ]
            if "public_ip_exposed" in types:
                attrs["associate_public_ip_address"] = False
            if "unencrypted_storage" in types:
                attrs["encrypted"] = True
                attrs["storage_encrypted"] = True
            if "overly_permissive_iam" in types:
                attrs["policy"] = {"Statement": []}
            if "default_credentials" in types:
                attrs["password_rotation_enabled"] = True
            if "unpatched_os" in types:
                attrs["patch_management"] = True
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
