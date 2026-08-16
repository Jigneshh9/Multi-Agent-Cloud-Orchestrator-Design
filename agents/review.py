"""Review Agent — static security/cost audit of a Terraform plan.

The review runs deterministic rules over the Terraform IR and produces typed
findings plus GitHub-style comments. It is then combined with the Red Team's
*dynamic* findings to form the final verdict.
"""

from __future__ import annotations

import json

from cloud_orchestra.agents.base import BaseAgent
from cloud_orchestra.schemas import (
    CommentCategory,
    FindingSeverity,
    ReviewComment,
    ReviewResult,
    ReviewVerdict,
    SecurityFinding,
    TerraformPlan,
)

_CVSS = {
    FindingSeverity.CRITICAL: 9.8,
    FindingSeverity.HIGH: 7.5,
    FindingSeverity.MEDIUM: 5.0,
    FindingSeverity.LOW: 2.5,
    FindingSeverity.INFO: 0.0,
}

_REMEDIATION = {
    "publicly_accessible_database": "Set publicly_accessible=false and restrict to a private subnet.",
    "open_ingress": "Restrict ingress CIDRs to internal ranges (e.g. 10.0.0.0/8).",
    "public_ip_exposed": "Disable public IP assignment; front the service with a load balancer.",
    "unencrypted_storage": "Enable encryption at rest (encrypted=true / SSE).",
    "overly_permissive_iam": "Scope IAM policies to least privilege (no wildcard actions).",
}


def static_scan(plan: TerraformPlan) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    for resource in plan.resources:
        attrs = resource.attributes
        rt = resource.resource_type.lower()

        if attrs.get("publicly_accessible") is True:
            findings.append(
                SecurityFinding(
                    attack_module="static_review",
                    vulnerability_type="publicly_accessible_database",
                    severity=FindingSeverity.CRITICAL,
                    target=resource.name,
                    description="Database is publicly reachable from the internet.",
                    remediation=_REMEDIATION["publicly_accessible_database"],
                    cvss_score=_CVSS[FindingSeverity.CRITICAL],
                )
            )

        ingress = attrs.get("ingress", [])
        for rule in ingress if isinstance(ingress, list) else []:
            if not isinstance(rule, dict):
                continue
            cidr_text = json.dumps(rule.get("cidr_blocks", rule.get("cidr", [])))
            if "0.0.0.0/0" in cidr_text or "::/0" in cidr_text:
                findings.append(
                    SecurityFinding(
                        attack_module="static_review",
                        vulnerability_type="open_ingress",
                        severity=FindingSeverity.HIGH,
                        target=resource.name,
                        description=f"Security group exposes port {rule.get('from_port', '*')} to the world.",
                        remediation=_REMEDIATION["open_ingress"],
                        cvss_score=_CVSS[FindingSeverity.HIGH],
                    )
                )

        if attrs.get("associate_public_ip_address") is True:
            findings.append(
                SecurityFinding(
                    attack_module="static_review",
                    vulnerability_type="public_ip_exposed",
                    severity=FindingSeverity.HIGH,
                    target=resource.name,
                    description="Instance is assigned a public IP address.",
                    remediation=_REMEDIATION["public_ip_exposed"],
                    cvss_score=_CVSS[FindingSeverity.HIGH],
                )
            )

        if "storage" in rt or "bucket" in rt or "volume" in rt or "disk" in rt or "db" in rt:
            encrypted = attrs.get("encrypted", attrs.get("storage_encrypted"))
            if encrypted is False:
                findings.append(
                    SecurityFinding(
                        attack_module="static_review",
                        vulnerability_type="unencrypted_storage",
                        severity=FindingSeverity.MEDIUM,
                        target=resource.name,
                        description="Data at rest is not encrypted.",
                        remediation=_REMEDIATION["unencrypted_storage"],
                        cvss_score=_CVSS[FindingSeverity.MEDIUM],
                    )
                )

        if "iam" in rt or "role" in rt:
            policy = attrs.get("policy", {})
            if isinstance(policy, dict):
                statements = policy.get("Statement", policy.get("statement", []))
                if not isinstance(statements, list):
                    statements = [statements]
                if any(
                    isinstance(s, dict) and s.get("Action") in ("*", ["*"]) for s in statements
                ):
                    findings.append(
                        SecurityFinding(
                            attack_module="static_review",
                            vulnerability_type="overly_permissive_iam",
                            severity=FindingSeverity.CRITICAL,
                            target=resource.name,
                            description="IAM policy grants wildcard actions.",
                            remediation=_REMEDIATION["overly_permissive_iam"],
                            cvss_score=_CVSS[FindingSeverity.CRITICAL],
                        )
                    )

    return findings


def _finding_to_comment(finding: SecurityFinding) -> ReviewComment:
    return ReviewComment(
        author="review-agent",
        category=CommentCategory.SECURITY,
        severity=finding.severity,
        body=f"[{finding.severity.value.upper()}] {finding.description} — {finding.remediation}",
    )


def merge_review(static: ReviewResult, dynamic: list[SecurityFinding]) -> ReviewResult:
    all_findings = static.findings + dynamic
    has_blocking = any(
        f.severity in (FindingSeverity.CRITICAL, FindingSeverity.HIGH) for f in all_findings
    )
    verdict = ReviewVerdict.CHANGES_REQUESTED if has_blocking else ReviewVerdict.APPROVED
    comments = list(static.comments) + [_finding_to_comment(f) for f in dynamic]
    return ReviewResult(
        verdict=verdict,
        comments=comments,
        findings=all_findings,
        cost_acceptable=static.cost_acceptable,
        security_acceptable=not has_blocking,
        summary=(
            f"{len(all_findings)} finding(s): "
            f"{sum(1 for f in all_findings if f.severity == FindingSeverity.CRITICAL)} critical, "
            f"{sum(1 for f in all_findings if f.severity == FindingSeverity.HIGH)} high."
        ),
    )


class ReviewAgent(BaseAgent):
    name = "review"

    async def review(self, plan: TerraformPlan, *, budget_usd: float | None = None) -> ReviewResult:
        async with self.timed(None, "static_review", input_summary=plan.description):
            findings = static_scan(plan)
            cost_acceptable = True
            cost_comments: list[ReviewComment] = []
            if budget_usd is not None and plan.estimated_monthly_cost_usd > budget_usd:
                cost_acceptable = False
                cost_comments.append(
                    ReviewComment(
                        author="review-agent",
                        category=CommentCategory.COST,
                        severity=FindingSeverity.MEDIUM,
                        body=(
                            f"Estimated ${plan.estimated_monthly_cost_usd:.2f}/mo exceeds "
                            f"budget ${budget_usd:.2f}/mo."
                        ),
                    )
                )

            comments = [_finding_to_comment(f) for f in findings] + cost_comments
            has_blocking = any(
                f.severity in (FindingSeverity.CRITICAL, FindingSeverity.HIGH) for f in findings
            )
            return ReviewResult(
                verdict=ReviewVerdict.CHANGES_REQUESTED if has_blocking else ReviewVerdict.APPROVED,
                comments=comments,
                findings=findings,
                cost_acceptable=cost_acceptable,
                security_acceptable=not has_blocking,
                summary=f"static scan: {len(findings)} finding(s)",
            )
