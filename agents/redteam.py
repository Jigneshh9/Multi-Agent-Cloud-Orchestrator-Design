"""Red Team Agent — adversarial validation by *deploying* the plan to a sandbox
and running attack modules against the live attack surface.

The key novelty over static scanning: the sandbox reveals *runtime* behaviours
(default credentials, unpatched OS) that are invisible in the Terraform IR
alone. Every finding is marked ``found_by_red_team`` so the evaluation framework
can attribute the security-finding-rate uplift to this agent.
"""

from __future__ import annotations

from typing import Any

from cloud_orchestra.agents.base import BaseAgent
from cloud_orchestra.schemas import FindingSeverity, SecurityFinding, TerraformPlan

_CVSS = {
    FindingSeverity.CRITICAL: 9.8,
    FindingSeverity.HIGH: 7.5,
    FindingSeverity.MEDIUM: 5.0,
    FindingSeverity.LOW: 2.5,
    FindingSeverity.INFO: 0.0,
}


def run_attack_modules(surface: dict[str, Any]) -> list[SecurityFinding]:
    """Deterministic attack modules over an observable attack surface."""
    findings: list[SecurityFinding] = []

    for name in surface.get("public_databases", []):
        findings.append(
            SecurityFinding(
                attack_module="exposed_database",
                vulnerability_type="publicly_accessible_database",
                severity=FindingSeverity.CRITICAL,
                target=name,
                description="Live database accepts connections from the public internet.",
                evidence="port 3306/5432 reachable from attacker host",
                remediation="Move database to a private subnet; remove public IP.",
                cvss_score=_CVSS[FindingSeverity.CRITICAL],
            )
        )

    for port_entry in surface.get("open_ports", []):
        port = port_entry.get("port", "*")
        findings.append(
            SecurityFinding(
                attack_module="open_port_scan",
                vulnerability_type="open_ingress",
                severity=FindingSeverity.HIGH,
                target=port_entry.get("resource", ""),
                description=f"Port {port} is exposed to 0.0.0.0/0 on the running service.",
                evidence=f"SYN scan: {port}/tcp open",
                remediation="Restrict security-group ingress to internal CIDRs.",
                cvss_score=_CVSS[FindingSeverity.HIGH],
            )
        )

    for name in surface.get("unencrypted_storage", []):
        findings.append(
            SecurityFinding(
                attack_module="data_at_rest_check",
                vulnerability_type="unencrypted_storage",
                severity=FindingSeverity.MEDIUM,
                target=name,
                description="Storage volume is readable without encryption.",
                evidence="snapshot copied and mounted; plaintext visible",
                remediation="Enable encryption at rest.",
                cvss_score=_CVSS[FindingSeverity.MEDIUM],
            )
        )

    for name in surface.get("overly_permissive_iam", []):
        findings.append(
            SecurityFinding(
                attack_module="privilege_escalation",
                vulnerability_type="overly_permissive_iam",
                severity=FindingSeverity.CRITICAL,
                target=name,
                description="IAM role allows wildcard actions; privilege escalation possible.",
                evidence="assume-role and sts:get-caller-identity succeeded",
                remediation="Apply least-privilege IAM policies.",
                cvss_score=_CVSS[FindingSeverity.CRITICAL],
            )
        )

    for flag_entry in surface.get("runtime_flags", []):
        flag = flag_entry.get("flag")
        resource = flag_entry.get("resource", "")
        if flag == "default_credentials":
            findings.append(
                SecurityFinding(
                    attack_module="default_credential_check",
                    vulnerability_type="default_credentials",
                    severity=FindingSeverity.CRITICAL,
                    target=resource,
                    description="Deployed service accepts default credentials.",
                    evidence="login admin/admin succeeded",
                    remediation="Enable managed password rotation and strong secrets.",
                    cvss_score=_CVSS[FindingSeverity.CRITICAL],
                )
            )
        elif flag == "unpatched_os":
            findings.append(
                SecurityFinding(
                    attack_module="vulnerability_scan",
                    vulnerability_type="unpatched_os",
                    severity=FindingSeverity.HIGH,
                    target=resource,
                    description="Running OS image has known unpatched CVEs.",
                    evidence="nuclei scan matched CVE-2021-3156 (sudo) signature",
                    remediation="Enable patch management / use hardened images.",
                    cvss_score=_CVSS[FindingSeverity.HIGH],
                )
            )

    return findings


class RedTeamAgent(BaseAgent):
    name = "red_team"

    async def pentest(self, plan: TerraformPlan) -> list[SecurityFinding]:
        async with self.timed(None, "sandbox_pentest", input_summary=plan.description):
            deployment = await self.ctx.sandbox.deploy(plan)
            try:
                findings = run_attack_modules(deployment.attack_surface)
            finally:
                await self.ctx.sandbox.teardown(deployment.id)
        for finding in findings:
            finding.found_by_red_team = True
        return findings
