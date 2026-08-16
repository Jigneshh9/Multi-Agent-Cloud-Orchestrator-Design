"""Evaluation harness — runs scenarios end-to-end and computes KPIs."""

from __future__ import annotations

from cloud_orchestra.core.config import FeatureFlags, Settings, get_settings
from cloud_orchestra.eval.metrics import EvalMetrics, compute_metrics
from cloud_orchestra.eval.scenarios import Scenario, build_scenarios
from cloud_orchestra.orchestrator.workflow import WorkflowResult
from cloud_orchestra.runtime import Runtime


class EvaluationHarness:
    def __init__(
        self,
        scenarios: list[Scenario] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.scenarios = scenarios or build_scenarios()
        self.settings = settings or get_settings()

    async def run(
        self, features: FeatureFlags | None = None
    ) -> tuple[list[WorkflowResult], EvalMetrics]:
        settings = self.settings if features is None else self.settings.with_features(features)
        runtime = Runtime(settings, persistent=False)
        results: list[WorkflowResult] = []
        try:
            for scenario in self.scenarios:
                results.append(await runtime.run_alert(scenario.alert))
        finally:
            await runtime.close()
        return results, compute_metrics(results)

    async def run_all_ablations(self) -> dict[str, EvalMetrics]:
        from cloud_orchestra.eval.ablations import ABLATIONS

        out: dict[str, EvalMetrics] = {}
        for name, features in ABLATIONS.items():
            _, metrics = await self.run(features)
            out[name] = metrics
        return out
