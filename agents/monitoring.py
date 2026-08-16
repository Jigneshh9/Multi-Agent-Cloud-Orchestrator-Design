"""Monitoring Agent — ingests raw AWS CloudWatch / GCP Monitoring / Azure Monitor
alerts and normalises them into the canonical :class:`Alert` contract."""

from __future__ import annotations

from typing import Any

from cloud_orchestra.agents.base import BaseAgent
from cloud_orchestra.core.events import Event, EventType
from cloud_orchestra.schemas import Alert, AlertSeverity, AlertSource, CloudProvider

_SEVERITY_MAP = {
    "CRITICAL": AlertSeverity.CRITICAL,
    "SEVERE": AlertSeverity.CRITICAL,
    "HIGH": AlertSeverity.HIGH,
    "MEDIUM": AlertSeverity.MEDIUM,
    "MODERATE": AlertSeverity.MEDIUM,
    "LOW": AlertSeverity.LOW,
    "INFO": AlertSeverity.INFO,
    "OK": AlertSeverity.INFO,
}


def _severity(raw: Any) -> AlertSeverity:
    if raw is None:
        return AlertSeverity.HIGH
    return _SEVERITY_MAP.get(str(raw).upper(), AlertSeverity.HIGH)


def _resource_type_from_name(name: str) -> str:
    n = name.lower()
    if "database" in n or "rds" in n or "sql" in n:
        return "rds_database"
    if "bucket" in n or "storage" in n:
        return "storage"
    if "loadbalancer" in n or "load-balancer" in n or "alb" in n:
        return "load_balancer"
    return "ec2_instance"


def parse_cloudwatch(payload: dict[str, Any]) -> Alert:
    alarm = payload.get("AlarmName") or payload.get("alarmName") or "aws-alarm"
    trigger = payload.get("Trigger") or {}
    metric = trigger.get("MetricName") or payload.get("MetricName") or "CPUUtilization"
    dimensions = trigger.get("Dimensions") or []
    resource_id = ""
    for dim in dimensions:
        if dim.get("name", "").lower() in ("instanceid", "instance_id"):
            resource_id = dim.get("value", "")
    return Alert(
        source=AlertSource.AWS_CLOUDWATCH,
        name=str(alarm),
        severity=_severity(payload.get("NewStateValue")),
        resource_type=_resource_type_from_name(str(alarm)),
        resource_id=resource_id or str(alarm),
        provider=CloudProvider.AWS,
        region=payload.get("Region") or payload.get("AWSAccountId"),
        metric_name=str(metric),
        threshold=float(trigger.get("Threshold", 80)) if trigger.get("Threshold") is not None else None,
        current_value=None,
        raw_payload=payload,
    )


def parse_gcp(payload: dict[str, Any]) -> Alert:
    incident = payload.get("incident") or {}
    metric = incident.get("metric") or {}
    name = str(incident.get("policy_name") or incident.get("resource_name") or "gcp-alarm")
    return Alert(
        source=AlertSource.GCP_MONITORING,
        name=name,
        severity=_severity(incident.get("state")),
        resource_type=_resource_type_from_name(name),
        resource_id=str(incident.get("resource_name") or name),
        provider=CloudProvider.GCP,
        region=incident.get("region"),
        metric_name=str(metric.get("type") or metric.get("displayName")),
        threshold=None,
        current_value=None,
        raw_payload=payload,
    )


def parse_azure(payload: dict[str, Any]) -> Alert:
    essentials = payload.get("essentials") or payload.get("data", {}).get("essentials") or {}
    name = str(essentials.get("alertRule") or "azure-alert")
    return Alert(
        source=AlertSource.AZURE_MONITOR,
        name=name,
        severity=_severity(essentials.get("severity")),
        resource_type=_resource_type_from_name(name),
        resource_id=str(essentials.get("alertTargetIDs") or name),
        provider=CloudProvider.AZURE,
        region=essentials.get("alertTargetIDs"),
        metric_name=str(essentials.get("monitorCondition")),
        threshold=None,
        current_value=None,
        raw_payload=payload,
    )


def parse_alert(source: AlertSource, payload: dict[str, Any]) -> Alert:
    if source == AlertSource.AWS_CLOUDWATCH:
        return parse_cloudwatch(payload)
    if source == AlertSource.GCP_MONITORING:
        return parse_gcp(payload)
    if source == AlertSource.AZURE_MONITOR:
        return parse_azure(payload)
    raise ValueError(f"unsupported alert source: {source}")


class MonitoringAgent(BaseAgent):
    name = "monitoring"

    async def ingest(self, alert: Alert) -> Alert:
        async with self.timed(None, "ingest", input_summary=alert.name):
            await self.ctx.repository.save_alert(alert)
            await self.ctx.bus.publish(
                Event(
                    type=EventType.ALERT_RECEIVED,
                    alert_id=alert.id,
                    agent=self.name,
                    payload={"name": alert.name, "severity": alert.severity.value},
                )
            )
        return alert

    async def ingest_raw(self, source: AlertSource, payload: dict[str, Any]) -> Alert:
        alert = parse_alert(source, payload)
        return await self.ingest(alert)
