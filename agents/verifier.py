"""Verifier Agent — closed-loop check that the applied Terraform actually
resolved the alert (novelty #1: self-healing verification)."""

from __future__ import annotations

from cloud_orchestra.agents.base import BaseAgent
from cloud_orchestra.providers.cloud import CloudClient
from cloud_orchestra.schemas import Alert, MetricComparison, TerraformPlan, VerificationResult


class VerifierAgent(BaseAgent):
    name = "verifier"

    async def verify(
        self, alert: Alert, plan: TerraformPlan, cloud: CloudClient
    ) -> VerificationResult:
        async with self.timed(None, "verify_resolution", input_summary=alert.name):
            metric_after = await cloud.apply_effect(alert, plan)
            metric_before = alert.current_value
            threshold = alert.threshold

            if alert.problem_class == "cost_anomaly":
                resolved = threshold is not None and metric_after <= threshold
            elif threshold is not None:
                resolved = metric_after < threshold
            else:
                resolved = metric_before is not None and metric_after < metric_before

            if resolved:
                comparison = MetricComparison.RESOLVED
            elif metric_before is not None and metric_after < metric_before:
                comparison = MetricComparison.PARTIAL
            else:
                comparison = MetricComparison.DEGRADED

            return VerificationResult(
                resolved=resolved,
                comparison=comparison,
                metric_before=metric_before,
                metric_after=metric_after,
                evidence={
                    "metric": alert.metric_name,
                    "threshold": threshold,
                    "before": metric_before,
                    "after": metric_after,
                },
                explanation=(
                    f"metric {alert.metric_name} moved from {metric_before} to {metric_after} "
                    f"(threshold {threshold}) -> {comparison.value}"
                ),
            )
