"""Orchestrator — the saga workflow coordinating all ten agents.

The workflow is a deterministic state machine; every step publishes an event on
the bus and records a decision trace, so each run is fully replayable. Feature
flags (from ``Settings.features``) let the evaluation framework disable each
novelty vector independently for the ablation studies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from cloud_orchestra.agents.base import AgentContext
from cloud_orchestra.agents.registry import AgentRegistry
from cloud_orchestra.agents.review import merge_review
from cloud_orchestra.core.events import Event, EventType
from cloud_orchestra.providers.terraform import render_hcl
from cloud_orchestra.schemas import (
    Alert,
    CloudProvider,
    MemoryEntry,
    PullRequest,
    ReviewResult,
    RollbackResult,
    Run,
    RunStatus,
    SecurityFinding,
    TerraformPlan,
    VerificationResult,
)

MAX_HARDEN_ITERATIONS = 3


@dataclass
class WorkflowResult:
    run: Run
    alert: Alert
    review: ReviewResult | None = None
    verification: VerificationResult | None = None
    rollback: RollbackResult | None = None
    pr: PullRequest | None = None
    explanation: str = ""
    findings: list[SecurityFinding] = field(default_factory=list)
    elapsed_ms: float = 0.0


class Orchestrator:
    def __init__(self, ctx: AgentContext, agents: AgentRegistry) -> None:
        self.ctx = ctx
        self.agents = agents
        self.features = ctx.settings.features

    async def run(self, alert: Alert) -> WorkflowResult:
        start = time.perf_counter()
        run = Run(
            alert_id=alert.id,
            trace_id=str(uuid4()),
            status=RunStatus.PENDING,
            ablation_config={
                "verifier": self.features.verifier,
                "rollback": self.features.rollback,
                "red_team": self.features.red_team,
                "fin_ops_rl": self.features.fin_ops_rl,
                "memory": self.features.memory,
                "cloud_harmonizer": self.features.cloud_harmonizer,
                "explainer": self.features.explainer,
            },
        )
        await self.ctx.repository.save_run(run)
        await self.ctx.bus.publish(
            Event(type=EventType.RUN_STARTED, run_id=run.id, alert_id=alert.id)
        )

        review: ReviewResult | None = None
        verification: VerificationResult | None = None
        rollback_result: RollbackResult | None = None
        pr: PullRequest | None = None
        explanation = ""
        findings: list[SecurityFinding] = []

        try:
            with self.ctx.tracer.context(run.id, run.trace_id):
                provider = await self._select_provider(run, alert)
                run.provider = provider

                memory_hits = await self._retrieve_memory(run, alert)

                run.status = RunStatus.PLANNING
                plan = await self.agents.devops.generate(alert, provider, memory_hits)
                run.terraform_plan = plan
                run.terraform_code = render_hcl(plan)
                run.cost_before = plan.estimated_monthly_cost_usd

                plan, review, detected = await self._review_and_harden(run, alert, plan)

                if self.features.fin_ops_rl:
                    run.status = RunStatus.COST_OPTIMIZING
                    plan = await self.agents.fin_ops.optimize(plan, alert, memory_hits)
                    run.cost_after = plan.estimated_monthly_cost_usd
                else:
                    run.cost_after = plan.estimated_monthly_cost_usd

                run.terraform_plan = plan
                run.terraform_code = render_hcl(plan)

                pr = await self._open_pr(run, alert, plan, review)

                run.status = RunStatus.APPLYING
                await self.ctx.terraform.apply(run.terraform_code)

                verification, rollback_result = await self._verify_and_rollback(run, alert, plan)
                await self._store_memory(run, alert, plan, verification)
                explanation = await self._explain(
                    run, alert, review, plan, verification, rollback_result, detected
                )

                findings = detected
                run.status = RunStatus.SUCCEEDED
        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error = str(exc)
            await self.ctx.bus.publish(
                Event(type=EventType.RUN_FAILED, run_id=run.id, payload={"error": str(exc)})
            )

        run.completed_at = datetime.now(UTC)
        await self.ctx.repository.save_run(run)
        await self.ctx.bus.publish(
            Event(type=EventType.RUN_COMPLETED, run_id=run.id, payload={"status": run.status.value})
        )

        return WorkflowResult(
            run=run,
            alert=alert,
            review=review,
            verification=verification,
            rollback=rollback_result,
            pr=pr,
            explanation=explanation,
            findings=findings,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )

    # -- steps ------------------------------------------------------------- #
    async def _select_provider(self, run: Run, alert: Alert) -> CloudProvider:
        if self.features.cloud_harmonizer:
            run.status = RunStatus.HARMONIZING
            recommendation = await self.agents.harmonizer.choose(alert)
            return recommendation.provider
        return alert.provider or CloudProvider.AWS

    async def _retrieve_memory(self, run: Run, alert: Alert) -> list[MemoryEntry]:
        if not self.features.memory:
            return []
        run.status = RunStatus.RETRIEVING_MEMORY
        return await self.agents.memory.retrieve(alert)

    async def _review_and_harden(
        self, run: Run, alert: Alert, plan: TerraformPlan
    ) -> tuple[TerraformPlan, ReviewResult, list[SecurityFinding]]:
        budget = alert.raw_payload.get("budget_usd")

        run.status = RunStatus.REVIEWING
        review = await self.agents.review.review(plan, budget_usd=budget)

        if self.features.red_team:
            run.status = RunStatus.RED_TEAMING
            red_findings = await self.agents.red_team.pentest(plan)
            review = merge_review(review, red_findings)

        detected = list(review.findings)

        current_plan = plan
        for _ in range(MAX_HARDEN_ITERATIONS):
            if review.security_acceptable:
                break
            current_plan = self.agents.devops.harden(current_plan, review.findings)
            review = await self.agents.review.review(current_plan, budget_usd=budget)
            if self.features.red_team:
                red_findings = await self.agents.red_team.pentest(current_plan)
                review = merge_review(review, red_findings)

        for finding in detected:
            finding.run_id = run.id
            await self.ctx.repository.save_finding(finding)
        for comment in review.comments:
            comment.run_id = run.id
            await self.ctx.repository.save_comment(comment)

        return current_plan, review, detected

    async def _open_pr(
        self, run: Run, alert: Alert, plan: TerraformPlan, review: ReviewResult | None
    ) -> PullRequest | None:
        run.status = RunStatus.OPENING_PR
        settings = self.ctx.settings
        if not settings.github_repo_owner or not settings.github_repo_name:
            return None
        pr = await self.ctx.github.create_pull_request(
            repo_owner=settings.github_repo_owner,
            repo_name=settings.github_repo_name,
            branch=f"cloud-orchestra/{run.id}",
            title=f"fix: remediate {alert.name}",
            body=plan.description,
            files={"main.tf": run.terraform_code},
        )
        if review and pr.pr_number is not None:
            for comment in review.comments:
                await self.ctx.github.add_comment(pr.pr_number, comment.body)
        await self.ctx.bus.publish(Event(type=EventType.PR_OPENED, run_id=run.id))
        return pr

    async def _verify_and_rollback(
        self, run: Run, alert: Alert, plan: TerraformPlan
    ) -> tuple[VerificationResult | None, RollbackResult | None]:
        if not self.features.verifier:
            run.resolved = True
            return None, None

        run.status = RunStatus.VERIFYING
        verification = await self.agents.verifier.verify(alert, plan, self.ctx.cloud)
        run.resolved = verification.resolved

        if not verification.resolved and self.features.rollback:
            run.status = RunStatus.ROLLING_BACK
            rollback_result = await self.agents.rollback.rollback(plan, self.ctx.terraform)
            run.status = RunStatus.ROLLED_BACK
            return verification, rollback_result
        return verification, None

    async def _store_memory(
        self,
        run: Run,
        alert: Alert,
        plan: TerraformPlan,
        verification: VerificationResult | None,
    ) -> None:
        if not self.features.memory:
            return
        cost_delta = (run.cost_before or 0.0) - (run.cost_after or 0.0)
        resolved = verification.resolved if verification else bool(run.resolved)
        await self.agents.memory.store(
            run, alert, plan, resolved=resolved, cost_delta_usd=cost_delta
        )
        await self.ctx.bus.publish(Event(type=EventType.MEMORY_STORED, run_id=run.id))

    async def _explain(
        self,
        run: Run,
        alert: Alert,
        review: ReviewResult | None,
        plan: TerraformPlan,
        verification: VerificationResult | None,
        rollback_result: RollbackResult | None,
        detected: list[SecurityFinding],
    ) -> str:
        if not self.features.explainer:
            return ""
        traces = self.ctx.tracer.for_run(run.id)
        return await self.agents.explainer.explain(
            run,
            alert,
            traces,
            review,
            plan,
            verification,
            rollback=rollback_result,
            findings=detected,
        )
