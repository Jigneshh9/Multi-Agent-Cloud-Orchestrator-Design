"""Sandbox adapter — deploys proposed infrastructure into an isolated sandbox so
the Red Team Agent can run live penetration tests, not just static scans.

* :class:`MockSandboxProvider` derives a deterministic *attack surface* from the
  Terraform IR (public IPs, open ports, public DBs, unencrypted storage, IAM).
* :class:`DockerSandboxProvider` (optional) would launch containers to exercise
  real network behaviour in production.
"""

from __future__ import annotations

import json
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from cloud_orchestra.core.errors import SandboxError
from cloud_orchestra.schemas import TerraformPlan


class SandboxDeployment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    endpoint: str = ""
    attack_surface: dict[str, Any] = Field(default_factory=dict)
    state: str = "running"


def derive_attack_surface(plan: TerraformPlan) -> dict[str, Any]:
    """Translate Terraform IR attributes into an observable attack surface."""
    from cloud_orchestra.providers.cloud import classify_resource

    surface: dict[str, Any] = {
        "public_endpoints": [],
        "open_ports": [],
        "public_databases": [],
        "unencrypted_storage": [],
        "overly_permissive_iam": [],
        "missing_security_groups": [],
        "services": [],
        "runtime_flags": [],
    }
    for resource in plan.resources:
        attrs = resource.attributes
        family = classify_resource(resource.resource_type)
        surface["services"].append(f"{resource.resource_type}.{resource.name}")

        if family == "database" and attrs.get("password_rotation_enabled") is False:
            surface["runtime_flags"].append(
                {"resource": resource.name, "flag": "default_credentials"}
            )
        if family in ("compute", "autoscaling") and attrs.get("patch_management") is False:
            surface["runtime_flags"].append(
                {"resource": resource.name, "flag": "unpatched_os"}
            )

        if attrs.get("publicly_accessible") is True:
            surface["public_databases"].append(resource.name)
        if attrs.get("associate_public_ip_address") is True or attrs.get("public_ip") is True:
            surface["public_endpoints"].append(resource.name)

        ingress = attrs.get("ingress", [])
        for rule in ingress if isinstance(ingress, list) else []:
            if not isinstance(rule, dict):
                continue
            cidrs = rule.get("cidr_blocks", rule.get("cidr", []))
            cidr_text = json.dumps(cidrs)
            if "0.0.0.0/0" in cidr_text or "::/0" in cidr_text:
                surface["open_ports"].append(
                    {
                        "resource": resource.name,
                        "port": rule.get("from_port", rule.get("port", "*")),
                    }
                )

        enc = attrs.get("encrypted")
        storage_enc = attrs.get("storage_encrypted")
        sse = attrs.get("server_side_encryption")
        if family in ("storage", "database") and (
            enc is False or storage_enc is False or (sse in (False, "None") and enc is None)
        ):
            surface["unencrypted_storage"].append(resource.name)

        if "iam" in resource.resource_type.lower() or "role" in resource.resource_type.lower():
            policy = attrs.get("policy", attrs.get("inline_policy", ""))
            if isinstance(policy, dict):
                statements = policy.get("Statement", policy.get("statement", []))
                if not isinstance(statements, list):
                    statements = [statements]
                for stmt in statements:
                    if isinstance(stmt, dict) and stmt.get("Action") in ("*", ["*"]):
                        surface["overly_permissive_iam"].append(resource.name)

        if family == "compute" and not attrs.get("security_groups"):
            surface["missing_security_groups"].append(resource.name)

    return surface


class SandboxProvider(Protocol):
    async def deploy(self, plan: TerraformPlan) -> SandboxDeployment: ...

    async def teardown(self, deployment_id: str) -> None: ...

    async def exec(self, deployment_id: str, command: str) -> str: ...


class MockSandboxProvider:
    """Deterministic sandbox that simulates deployment from the Terraform IR."""

    def __init__(self) -> None:
        self._deployments: dict[str, SandboxDeployment] = {}

    async def deploy(self, plan: TerraformPlan) -> SandboxDeployment:
        deployment = SandboxDeployment(
            endpoint="sandbox://mock",
            attack_surface=derive_attack_surface(plan),
        )
        self._deployments[deployment.id] = deployment
        return deployment

    async def teardown(self, deployment_id: str) -> None:
        self._deployments.pop(deployment_id, None)

    async def exec(self, deployment_id: str, command: str) -> str:
        if deployment_id not in self._deployments:
            raise SandboxError(f"unknown deployment: {deployment_id}")
        # Commands such as "nmap" return nothing in the mock; the attack
        # modules instead read the deployment's attack surface directly.
        return ""


class DockerSandboxProvider:
    """Optional production sandbox using Docker (not exercised in unit tests)."""

    def __init__(self, network: str = "cloud-orchestra-sandbox") -> None:
        self._network = network

    async def deploy(self, plan: TerraformPlan) -> SandboxDeployment:
        raise SandboxError("DockerSandboxProvider requires a container runtime and is opt-in")

    async def teardown(self, deployment_id: str) -> None:
        return None

    async def exec(self, deployment_id: str, command: str) -> str:
        raise SandboxError("DockerSandboxProvider exec is opt-in")


def build_sandbox_provider(provider: str) -> SandboxProvider:
    if provider == "docker":
        return DockerSandboxProvider()
    return MockSandboxProvider()
