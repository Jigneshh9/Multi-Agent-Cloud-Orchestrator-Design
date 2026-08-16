"""SQLAlchemy ORM models.

Values are persisted with string UUIDs and JSON-serialised text columns so the
same schema works on SQLite (dev/tests) and PostgreSQL (production) without
dialect-specific types.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _pk() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class AlertRow(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_pk)
    source: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(255))
    severity: Mapped[str] = mapped_column(String(32))
    resource_type: Mapped[str] = mapped_column(String(128))
    resource_id: Mapped[str] = mapped_column(String(255))
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    region: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metric_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    fired_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    raw_payload: Mapped[str] = mapped_column(Text, default="{}")


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_pk)
    alert_id: Mapped[str] = mapped_column(String(36), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    terraform_plan: Mapped[str] = mapped_column(Text, default="")
    terraform_code: Mapped[str] = mapped_column(Text, default="")
    review_verdict: Mapped[str | None] = mapped_column(String(32), nullable=True)
    cost_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    metric_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    metric_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolved: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    trace_id: Mapped[str] = mapped_column(String(64), default="")
    ablation_config: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class DecisionTraceRow(Base):
    __tablename__ = "decision_traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_pk)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    trace_id: Mapped[str] = mapped_column(String(64), default="")
    agent: Mapped[str] = mapped_column(String(64))
    step: Mapped[str] = mapped_column(String(128))
    input_summary: Mapped[str] = mapped_column(Text, default="")
    output_summary: Mapped[str] = mapped_column(Text, default="")
    rationale: Mapped[str] = mapped_column(Text, default="")
    llm_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReviewCommentRow(Base):
    __tablename__ = "review_comments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_pk)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    author: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(32))
    severity: Mapped[str] = mapped_column(String(32), default="info")
    path: Mapped[str] = mapped_column(String(255), default="")
    line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    body: Mapped[str] = mapped_column(Text, default="")


class SecurityFindingRow(Base):
    __tablename__ = "security_findings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_pk)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    attack_module: Mapped[str] = mapped_column(String(128))
    vulnerability_type: Mapped[str] = mapped_column(String(128))
    severity: Mapped[str] = mapped_column(String(32))
    target: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[str] = mapped_column(Text, default="")
    remediation: Mapped[str] = mapped_column(Text, default="")
    cvss_score: Mapped[float] = mapped_column(Float, default=0.0)
    reproducible: Mapped[bool] = mapped_column(Boolean, default=True)
    found_by_red_team: Mapped[bool] = mapped_column(Boolean, default=False)


class MemoryEntryRow(Base):
    __tablename__ = "memory_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_pk)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    problem_class: Mapped[str] = mapped_column(String(64), index=True)
    provider: Mapped[str] = mapped_column(String(32))
    resource_type: Mapped[str] = mapped_column(String(128))
    summary: Mapped[str] = mapped_column(Text, default="")
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    cost_delta_usd: Mapped[float] = mapped_column(Float, default=0.0)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EvaluationRunRow(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_pk)
    scenario_id: Mapped[str] = mapped_column(String(64))
    ablation: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tsr: Mapped[float] = mapped_column(Float, default=0.0)
    sfr: Mapped[float] = mapped_column(Float, default=0.0)
    cost_savings: Mapped[float] = mapped_column(Float, default=0.0)
    mttr: Mapped[float] = mapped_column(Float, default=0.0)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
