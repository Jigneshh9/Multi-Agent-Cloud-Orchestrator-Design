"""Repository — bridges Pydantic domain objects and SQLAlchemy rows."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cloud_orchestra.core.tracing import DecisionTrace
from cloud_orchestra.db.models import (
    AlertRow,
    DecisionTraceRow,
    EvaluationRunRow,
    MemoryEntryRow,
    ReviewCommentRow,
    RunRow,
    SecurityFindingRow,
)
from cloud_orchestra.schemas import (
    Alert,
    AlertSeverity,
    AlertSource,
    CloudProvider,
    MemoryEntry,
    ReviewComment,
    ReviewResult,
    Run,
    RunStatus,
    SecurityFinding,
    TerraformPlan,
)


def _json_dump(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


class Repository:
    """Async repository. A single session is used for a single workflow run."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def commit(self) -> None:
        await self._session.commit()

    # -- alerts ------------------------------------------------------------ #
    async def save_alert(self, alert: Alert) -> Alert:
        row = AlertRow(
            id=str(alert.id),
            source=alert.source.value,
            name=alert.name,
            severity=alert.severity.value,
            resource_type=alert.resource_type,
            resource_id=alert.resource_id,
            provider=alert.provider.value if alert.provider else None,
            region=alert.region,
            metric_name=alert.metric_name,
            threshold=alert.threshold,
            current_value=alert.current_value,
            fired_at=alert.fired_at,
            raw_payload=_json_dump(alert.raw_payload),
        )
        self._session.add(row)
        await self._session.flush()
        return alert

    async def get_alert(self, alert_id: UUID) -> Alert | None:
        row = await self._session.get(AlertRow, str(alert_id))
        return self._alert_from_row(row) if row else None

    # -- runs -------------------------------------------------------------- #
    async def save_run(self, run: Run) -> Run:
        row = await self._session.get(RunRow, str(run.id))
        if row is None:
            row = RunRow(id=str(run.id), alert_id=str(run.alert_id))
            self._session.add(row)
        self._apply_run(row, run)
        await self._session.flush()
        return run

    async def get_run(self, run_id: UUID) -> Run | None:
        row = await self._session.get(RunRow, str(run_id))
        return self._run_from_row(row) if row else None

    async def list_runs(self, status: RunStatus | None = None) -> list[Run]:
        stmt = select(RunRow)
        if status is not None:
            stmt = stmt.where(RunRow.status == status.value)
        result = await self._session.execute(stmt.order_by(RunRow.created_at))
        return [self._run_from_row(r) for r in result.scalars()]

    # -- traces ------------------------------------------------------------ #
    async def save_trace(self, trace: DecisionTrace) -> None:
        self._session.add(
            DecisionTraceRow(
                id=str(trace.id),
                run_id=str(trace.run_id) if trace.run_id else "",
                trace_id=trace.trace_id,
                agent=trace.agent,
                step=trace.step,
                input_summary=trace.input_summary,
                output_summary=trace.output_summary,
                rationale=trace.rationale,
                llm_model=trace.llm_model,
                input_tokens=trace.input_tokens,
                output_tokens=trace.output_tokens,
                latency_ms=trace.latency_ms,
                metadata_json=_json_dump(trace.metadata),
                timestamp=trace.timestamp,
            )
        )
        await self._session.flush()

    async def traces_for_run(self, run_id: UUID) -> list[DecisionTrace]:
        result = await self._session.execute(
            select(DecisionTraceRow)
            .where(DecisionTraceRow.run_id == str(run_id))
            .order_by(DecisionTraceRow.timestamp)
        )
        return [self._trace_from_row(r) for r in result.scalars()]

    # -- findings / comments ----------------------------------------------- #
    async def save_finding(self, finding: SecurityFinding) -> None:
        self._session.add(
            SecurityFindingRow(
                id=str(finding.id),
                run_id=str(finding.run_id) if finding.run_id else "",
                attack_module=finding.attack_module,
                vulnerability_type=finding.vulnerability_type,
                severity=finding.severity.value,
                target=finding.target,
                description=finding.description,
                evidence=finding.evidence,
                remediation=finding.remediation,
                cvss_score=finding.cvss_score,
                reproducible=finding.reproducible,
                found_by_red_team=finding.found_by_red_team,
            )
        )
        await self._session.flush()

    async def save_comment(self, comment: ReviewComment) -> None:
        self._session.add(
            ReviewCommentRow(
                id=str(comment.id),
                run_id=str(comment.run_id) if comment.run_id else "",
                author=comment.author,
                category=comment.category.value,
                severity=comment.severity.value,
                path=comment.path,
                line=comment.line,
                body=comment.body,
            )
        )
        await self._session.flush()

    # -- memory ------------------------------------------------------------ #
    async def save_memory(self, entry: MemoryEntry) -> None:
        self._session.add(
            MemoryEntryRow(
                id=str(entry.id),
                run_id=str(entry.run_id) if entry.run_id else "",
                problem_class=entry.problem_class,
                provider=entry.provider.value,
                resource_type=entry.resource_type,
                summary=entry.summary,
                success=entry.success,
                cost_delta_usd=entry.cost_delta_usd,
                resolved=entry.resolved,
                latency_ms=entry.latency_ms,
                created_at=entry.created_at,
            )
        )
        await self._session.flush()

    async def query_memory(self, problem_class: str | None = None) -> list[MemoryEntry]:
        stmt = select(MemoryEntryRow).order_by(MemoryEntryRow.created_at.desc())
        if problem_class:
            stmt = stmt.where(MemoryEntryRow.problem_class == problem_class)
        result = await self._session.execute(stmt)
        return [self._memory_from_row(r) for r in result.scalars()]

    # -- evaluation -------------------------------------------------------- #
    async def save_evaluation(
        self,
        scenario_id: str,
        metrics: dict[str, float],
        *,
        ablation: str | None = None,
    ) -> None:
        self._session.add(
            EvaluationRunRow(
                scenario_id=scenario_id,
                ablation=ablation,
                tsr=metrics.get("tsr", 0.0),
                sfr=metrics.get("sfr", 0.0),
                cost_savings=metrics.get("cost_savings", 0.0),
                mttr=metrics.get("mttr", 0.0),
                metrics_json=json.dumps(metrics),
            )
        )
        await self._session.flush()

    # -- mapping helpers --------------------------------------------------- #
    @staticmethod
    def _apply_run(row: RunRow, run: Run) -> None:
        row.status = run.status.value
        row.provider = run.provider.value if run.provider else None
        row.terraform_plan = (
            run.terraform_plan.model_dump_json() if run.terraform_plan else ""
        )
        row.terraform_code = run.terraform_code
        row.review_verdict = (
            run.review_result.verdict.value if run.review_result else None
        )
        row.cost_before = run.cost_before
        row.cost_after = run.cost_after
        row.metric_before = run.metric_before
        row.metric_after = run.metric_after
        row.resolved = run.resolved
        row.error = run.error
        row.trace_id = run.trace_id
        row.ablation_config = json.dumps(run.ablation_config)
        row.completed_at = run.completed_at

    @staticmethod
    def _alert_from_row(row: AlertRow) -> Alert:
        return Alert(
            id=UUID(row.id),
            source=AlertSource(row.source),
            name=row.name,
            severity=AlertSeverity(row.severity),
            resource_type=row.resource_type,
            resource_id=row.resource_id,
            provider=CloudProvider(row.provider) if row.provider else None,
            region=row.region,
            metric_name=row.metric_name,
            threshold=row.threshold,
            current_value=row.current_value,
            fired_at=row.fired_at,
            raw_payload=json.loads(row.raw_payload or "{}"),
        )

    @staticmethod
    def _run_from_row(row: RunRow) -> Run:
        return Run(
            id=UUID(row.id),
            alert_id=UUID(row.alert_id),
            status=RunStatus(row.status),
            provider=CloudProvider(row.provider) if row.provider else None,
            terraform_plan=(
                TerraformPlan.model_validate_json(row.terraform_plan)
                if row.terraform_plan
                else None
            ),
            terraform_code=row.terraform_code,
            review_result=(
                ReviewResult(verdict=row.review_verdict) if row.review_verdict else None
            ),
            cost_before=row.cost_before,
            cost_after=row.cost_after,
            metric_before=row.metric_before,
            metric_after=row.metric_after,
            resolved=row.resolved,
            error=row.error,
            trace_id=row.trace_id,
            ablation_config=json.loads(row.ablation_config or "{}"),
            created_at=row.created_at,
            updated_at=row.updated_at,
            completed_at=row.completed_at,
        )

    @staticmethod
    def _trace_from_row(row: DecisionTraceRow) -> DecisionTrace:
        return DecisionTrace(
            id=UUID(row.id),
            run_id=UUID(row.run_id) if row.run_id else None,
            trace_id=row.trace_id,
            agent=row.agent,
            step=row.step,
            input_summary=row.input_summary,
            output_summary=row.output_summary,
            rationale=row.rationale,
            llm_model=row.llm_model,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            latency_ms=row.latency_ms,
            metadata=json.loads(row.metadata_json or "{}"),
            timestamp=row.timestamp,
        )

    @staticmethod
    def _memory_from_row(row: MemoryEntryRow) -> MemoryEntry:
        return MemoryEntry(
            id=UUID(row.id),
            run_id=UUID(row.run_id) if row.run_id else None,
            problem_class=row.problem_class,
            provider=CloudProvider(row.provider),
            resource_type=row.resource_type,
            summary=row.summary,
            success=row.success,
            cost_delta_usd=row.cost_delta_usd,
            resolved=row.resolved,
            latency_ms=row.latency_ms,
            created_at=row.created_at,
        )


class InMemoryRepository(Repository):
    """In-memory repository used by the evaluation harness (no DB required)."""

    def __init__(self) -> None:
        # Bypass the SQLAlchemy-backed parent.
        self._alerts: dict[str, Alert] = {}
        self._runs: dict[str, Run] = {}
        self._traces: list[DecisionTrace] = []
        self._findings: list[SecurityFinding] = []
        self._comments: list[ReviewComment] = []
        self._memory: list[MemoryEntry] = []

    async def commit(self) -> None:
        return None

    async def save_alert(self, alert: Alert) -> Alert:
        self._alerts[str(alert.id)] = alert
        return alert

    async def get_alert(self, alert_id: UUID) -> Alert | None:
        return self._alerts.get(str(alert_id))

    async def save_run(self, run: Run) -> Run:
        self._runs[str(run.id)] = run
        return run

    async def get_run(self, run_id: UUID) -> Run | None:
        return self._runs.get(str(run_id))

    async def list_runs(self, status: RunStatus | None = None) -> list[Run]:
        runs = list(self._runs.values())
        if status is not None:
            runs = [r for r in runs if r.status == status]
        return runs

    async def save_trace(self, trace: DecisionTrace) -> None:
        self._traces.append(trace)

    async def traces_for_run(self, run_id: UUID) -> list[DecisionTrace]:
        return [t for t in self._traces if t.run_id == run_id]

    async def save_finding(self, finding: SecurityFinding) -> None:
        self._findings.append(finding)

    async def save_comment(self, comment: ReviewComment) -> None:
        self._comments.append(comment)

    async def save_memory(self, entry: MemoryEntry) -> None:
        self._memory.append(entry)

    async def query_memory(self, problem_class: str | None = None) -> list[MemoryEntry]:
        if problem_class is None:
            return list(self._memory)
        return [m for m in self._memory if m.problem_class == problem_class]

    async def save_evaluation(
        self,
        scenario_id: str,
        metrics: dict[str, float],
        *,
        ablation: str | None = None,
    ) -> None:
        # Evaluations are aggregated in the harness; no-op here.
        return None

    @property
    def findings(self) -> list[SecurityFinding]:
        return list(self._findings)

    @property
    def comments(self) -> list[ReviewComment]:
        return list(self._comments)
