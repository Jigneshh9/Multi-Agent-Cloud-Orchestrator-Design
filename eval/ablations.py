"""The five ablation studies.

Each study disables exactly one novelty vector and re-runs the full scenario
suite so the resulting KPI deltas isolate that vector's contribution:

1. ``ablate_closed_loop``   — verifier + rollback off (self-healing).
2. ``ablate_adversarial``   — red team off (adversarial validation).
3. ``ablate_rl_finops``     — RL off, greedy baseline (FinOps).
4. ``ablate_memory``        — RAG memory off (persistent memory).
5. ``ablate_harmonizer``    — cloud harmonizer off (multi-cloud).

Explainability (novelty #5) is validated intrinsically via
``explanation_coverage`` in the baseline metrics rather than an ablation.
"""

from __future__ import annotations

from cloud_orchestra.core.config import FeatureFlags
from cloud_orchestra.eval.metrics import EvalMetrics, metrics_table

ABLATIONS: dict[str, FeatureFlags] = {
    "baseline": FeatureFlags(),
    "ablate_closed_loop": FeatureFlags(verifier=False, rollback=False),
    "ablate_adversarial": FeatureFlags(red_team=False),
    "ablate_rl_finops": FeatureFlags(fin_ops_rl=False),
    "ablate_memory": FeatureFlags(memory=False),
    "ablate_harmonizer": FeatureFlags(cloud_harmonizer=False),
}


def format_report(results: dict[str, EvalMetrics]) -> str:
    ordered = ["baseline"] + [k for k in results if k != "baseline"]
    rows = [(name, results[name]) for name in ordered if name in results]
    return metrics_table(rows)


def compute_deltas(results: dict[str, EvalMetrics]) -> dict[str, dict[str, float]]:
    base = results.get("baseline")
    if base is None:
        return {}
    deltas: dict[str, dict[str, float]] = {}
    for name, m in results.items():
        if name == "baseline":
            continue
        deltas[name] = {
            "tsr_delta": round(m.tsr - base.tsr, 4),
            "sfr_delta": round(m.sfr - base.sfr, 4),
            "cost_savings_delta": round(m.cost_savings - base.cost_savings, 4),
            "mttr_delta": round(m.mttr - base.mttr, 4),
        }
    return deltas
