"""Evaluation metrics for Cloud-Orchestra.

The four headline KPIs computed over a batch of workflow runs:

* **TSR** — Task Success Rate: fraction of runs whose alert was resolved.
* **SFR** — Security Finding Rate: mean number of security findings detected
  per run (static + adversarial).
* **Cost Savings** — mean relative monthly-cost reduction vs. the DevOps plan.
* **MTTR** — Mean Time To Remediation (seconds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cloud_orchestra.orchestrator.workflow import WorkflowResult


@dataclass
class EvalMetrics:
    tsr: float = 0.0
    sfr: float = 0.0
    cost_savings: float = 0.0
    mttr: float = 0.0
    n: int = 0
    explanation_coverage: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        return {
            "tsr": round(self.tsr, 4),
            "sfr": round(self.sfr, 4),
            "cost_savings": round(self.cost_savings, 4),
            "mttr": round(self.mttr, 4),
            "n": float(self.n),
            "explanation_coverage": round(self.explanation_coverage, 4),
        }


def compute_metrics(results: list[WorkflowResult]) -> EvalMetrics:
    n = len(results)
    if n == 0:
        return EvalMetrics()

    resolved = sum(1 for r in results if bool(r.run.resolved))
    findings = sum(len(r.findings) for r in results)
    savings = [
        _cost_savings(r)
        for r in results
        if r.run.cost_before is not None and r.run.cost_before > 0
    ]
    mttr = sum(r.elapsed_ms for r in results) / 1000.0 / n
    explained = sum(1 for r in results if r.explanation) / n

    return EvalMetrics(
        tsr=resolved / n,
        sfr=findings / n,
        cost_savings=(sum(savings) / len(savings)) if savings else 0.0,
        mttr=mttr,
        n=n,
        explanation_coverage=explained,
        extra={"resolved": resolved, "total_findings": findings},
    )


def _cost_savings(result: WorkflowResult) -> float:
    before = result.run.cost_before or 0.0
    after = result.run.cost_after or 0.0
    if before <= 0:
        return 0.0
    return max(0.0, (before - after) / before)


def metrics_table(rows: list[tuple[str, EvalMetrics]]) -> str:
    header = f"{'config':<28} {'TSR':>8} {'SFR':>8} {'CostSav':>8} {'MTTR(s)':>9} {'ExplCov':>8}"
    lines = [header, "-" * len(header)]
    for name, m in rows:
        lines.append(
            f"{name:<28} {m.tsr:>8.3f} {m.sfr:>8.3f} {m.cost_savings:>8.3f} "
            f"{m.mttr:>9.2f} {m.explanation_coverage:>8.3f}"
        )
    return "\n".join(lines)
