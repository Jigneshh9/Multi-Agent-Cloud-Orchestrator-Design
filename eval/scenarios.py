"""Evaluation scenarios — golden (alert -> expected outcome) pairs."""

from __future__ import annotations

from dataclasses import dataclass

from cloud_orchestra.schemas import Alert, AlertSeverity, AlertSource, CloudProvider


@dataclass
class Scenario:
    id: str
    alert: Alert
    expect_resolved: bool = True
    expect_min_findings: int = 1
    budget_usd: float | None = None


def _alert(
    name: str,
    *,
    resource_type: str,
    current: float | None,
    threshold: float | None,
    provider: CloudProvider | None = None,
    region: str = "us-east-1",
    budget: float | None = None,
) -> Alert:
    raw: dict[str, object] = {}
    if budget is not None:
        raw["budget_usd"] = budget
    return Alert(
        source=AlertSource.AWS_CLOUDWATCH,
        name=name,
        severity=AlertSeverity.HIGH,
        resource_type=resource_type,
        resource_id=f"{resource_type}-001",
        provider=provider,
        region=region,
        metric_name="CPUUtilization",
        threshold=threshold,
        current_value=current,
        raw_payload=raw,
    )


def build_scenarios() -> list[Scenario]:
    return [
        Scenario(
            id="high_cpu",
            alert=_alert("High CPU on web tier", resource_type="ec2_instance",
                          current=90.0, threshold=80.0),
            expect_resolved=True,
            expect_min_findings=2,  # open ingress + public IP (static & dynamic)
        ),
        Scenario(
            id="db_capacity",
            alert=_alert("DB connection pool exhausted", resource_type="rds_database",
                          current=95.0, threshold=85.0),
            expect_resolved=True,
            expect_min_findings=2,  # public DB + default credentials (runtime)
        ),
        Scenario(
            id="storage_full",
            alert=_alert("Storage volume full", resource_type="storage",
                          current=98.0, threshold=90.0),
            expect_resolved=True,
            expect_min_findings=1,  # unencrypted storage
        ),
        Scenario(
            id="high_memory",
            alert=_alert("Memory pressure on worker", resource_type="ec2_instance",
                          current=92.0, threshold=80.0, provider=CloudProvider.GCP,
                          region="eu-west-1"),
            expect_resolved=True,
            expect_min_findings=0,
        ),
        Scenario(
            id="high_latency",
            alert=_alert("P95 latency breach", resource_type="load_balancer",
                          current=1200.0, threshold=800.0, provider=CloudProvider.AZURE,
                          region="eu-central-1"),
            expect_resolved=True,
            expect_min_findings=0,
        ),
        Scenario(
            id="cost_anomaly",
            alert=_alert("Monthly spend anomaly", resource_type="ec2_instance",
                          current=2500.0, threshold=1500.0, budget=1500.0),
            expect_resolved=True,
            expect_min_findings=0,
        ),
    ]
