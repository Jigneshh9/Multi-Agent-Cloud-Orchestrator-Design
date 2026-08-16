"""Shared Pydantic domain contracts (schemas).

These are the single source of truth for every agent-to-agent message, API
request/response and the Terraform intermediate representation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def new_id() -> UUID:
    return uuid4()


def utcnow() -> datetime:
    return datetime.utcnow()


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class AlertSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertSource(StrEnum):
    AWS_CLOUDWATCH = "aws_cloudwatch"
    GCP_MONITORING = "gcp_monitoring"
    AZURE_MONITOR = "azure_monitor"


class CloudProvider(StrEnum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class RunStatus(StrEnum):
    PENDING = "pending"
    HARMONIZING = "harmonizing"
    RETRIEVING_MEMORY = "retrieving_memory"
    PLANNING = "planning"
    REVIEWING = "reviewing"
    RED_TEAMING = "red_teaming"
    COST_OPTIMIZING = "cost_optimizing"
    OPENING_PR = "opening_pr"
    APPLYING = "applying"
    VERIFYING = "verifying"
    ROLLING_BACK = "rolling_back"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ABORTED = "aborted"


class ReviewVerdict(StrEnum):
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    REJECTED = "rejected"


class FindingSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class CommentCategory(StrEnum):
    SECURITY = "security"
    COST = "cost"
    FUNCTIONAL = "functional"
    STYLE = "style"
    EXPLANATION = "explanation"


class MetricComparison(StrEnum):
    RESOLVED = "resolved"
    DEGRADED = "degraded"
    PARTIAL = "partial"


# --------------------------------------------------------------------------- #
# Alert
# --------------------------------------------------------------------------- #
class Alert(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: UUID = Field(default_factory=new_id)
    source: AlertSource
    name: str
    severity: AlertSeverity
    resource_type: str
    resource_id: str
    provider: CloudProvider | None = None
    region: str | None = None
    metric_name: str | None = None
    threshold: float | None = None
    current_value: float | None = None
    fired_at: datetime = Field(default_factory=utcnow)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def problem_class(self) -> str:
        """Normalised remediation category used for memory retrieval."""
        name = self.name.lower()
        if "cpu" in name or "utilization" in name:
            return "high_cpu"
        if "memory" in name:
            return "high_memory"
        if "storage" in name or "disk" in name:
            return "storage_full"
        if "connection" in name or "pool" in name or "database" in name:
            return "db_capacity"
        if "latency" in name:
            return "high_latency"
        if "cost" in name or "spend" in name or "budget" in name:
            return "cost_anomaly"
        return "generic"


# --------------------------------------------------------------------------- #
# Terraform intermediate representation (IR)
# --------------------------------------------------------------------------- #
class TerraformResource(BaseModel):
    resource_type: str  # e.g. aws_instance, google_compute_instance
    name: str
    provider: CloudProvider
    attributes: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class TerraformPlan(BaseModel):
    provider: CloudProvider
    provider_config: dict[str, Any] = Field(default_factory=dict)
    resources: list[TerraformResource] = Field(default_factory=list)
    variables: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    estimated_monthly_cost_usd: float = 0.0
    latency_ms: int = 0
    tags: dict[str, str] = Field(default_factory=dict)

    def resource_by_type(self, resource_type: str) -> list[TerraformResource]:
        return [r for r in self.resources if r.resource_type == resource_type]


# --------------------------------------------------------------------------- #
# Findings / comments
# --------------------------------------------------------------------------- #
class SecurityFinding(BaseModel):
    id: UUID = Field(default_factory=new_id)
    run_id: UUID | None = None
    attack_module: str
    vulnerability_type: str
    severity: FindingSeverity
    target: str
    description: str
    evidence: str = ""
    remediation: str = ""
    cvss_score: float = 0.0
    reproducible: bool = True
    found_by_red_team: bool = False


class ReviewComment(BaseModel):
    id: UUID = Field(default_factory=new_id)
    run_id: UUID | None = None
    author: str
    category: CommentCategory
    severity: FindingSeverity = FindingSeverity.INFO
    path: str = ""
    line: int | None = None
    body: str


class ReviewResult(BaseModel):
    verdict: ReviewVerdict
    comments: list[ReviewComment] = Field(default_factory=list)
    findings: list[SecurityFinding] = Field(default_factory=list)
    cost_acceptable: bool = True
    security_acceptable: bool = True
    summary: str = ""


# --------------------------------------------------------------------------- #
# Verification / application results
# --------------------------------------------------------------------------- #
class ApplyResult(BaseModel):
    succeeded: bool
    plan_output: str = ""
    apply_output: str = ""
    applied_resources: list[str] = Field(default_factory=list)
    error: str = ""


class VerificationResult(BaseModel):
    resolved: bool
    comparison: MetricComparison
    metric_before: float | None = None
    metric_after: float | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""


class RollbackResult(BaseModel):
    succeeded: bool
    reverted_resources: list[str] = Field(default_factory=list)
    error: str = ""


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
class MemoryEntry(BaseModel):
    id: UUID = Field(default_factory=new_id)
    run_id: UUID | None = None
    problem_class: str
    provider: CloudProvider
    resource_type: str
    summary: str
    terraform_plan: TerraformPlan | None = None
    success: bool = False
    cost_delta_usd: float = 0.0
    resolved: bool = False
    latency_ms: int = 0
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# PR
# --------------------------------------------------------------------------- #
class PullRequest(BaseModel):
    id: UUID = Field(default_factory=new_id)
    run_id: UUID | None = None
    repo_owner: str
    repo_name: str
    branch: str
    title: str
    body: str
    pr_number: int | None = None
    pr_url: str = ""
    status: Literal["created", "simulated", "failed"] = "simulated"


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #
class ProviderRecommendation(BaseModel):
    provider: CloudProvider
    score: float
    reasoning: list[str] = Field(default_factory=list)
    alternatives: list[tuple[CloudProvider, float]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Run (orchestrator state)
# --------------------------------------------------------------------------- #
class Run(BaseModel):
    id: UUID = Field(default_factory=new_id)
    alert_id: UUID
    status: RunStatus = RunStatus.PENDING
    provider: CloudProvider | None = None
    terraform_plan: TerraformPlan | None = None
    terraform_code: str = ""
    review_result: ReviewResult | None = None
    cost_before: float | None = None
    cost_after: float | None = None
    metric_before: float | None = None
    metric_after: float | None = None
    resolved: bool | None = None
    error: str | None = None
    trace_id: str = ""
    ablation_config: dict[str, bool] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
