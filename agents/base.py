"""Agent base classes and the shared runtime context."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from cloud_orchestra.core.bus import EventBus
from cloud_orchestra.core.config import Settings
from cloud_orchestra.core.llm import LLMClient
from cloud_orchestra.core.metrics import MetricsRegistry
from cloud_orchestra.core.tracing import Tracer
from cloud_orchestra.db.repository import Repository
from cloud_orchestra.memory.store import MemoryStore
from cloud_orchestra.providers.cloud import CloudClient
from cloud_orchestra.providers.github import GitHubClient
from cloud_orchestra.providers.sandbox import SandboxProvider
from cloud_orchestra.providers.terraform import TerraformProvider


@dataclass
class AgentContext:
    """Shared services available to every agent."""

    settings: Settings
    llm: LLMClient
    tracer: Tracer
    metrics: MetricsRegistry
    repository: Repository
    memory: MemoryStore
    cloud: CloudClient
    github: GitHubClient
    sandbox: SandboxProvider
    terraform: TerraformProvider
    bus: EventBus
    finops_policy: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """Common behaviour: shared context, decision tracing and timing."""

    name: str = "base"

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    def trace(
        self,
        run_id: UUID | None,
        step: str,
        *,
        input_summary: str = "",
        output_summary: str = "",
        rationale: str = "",
        trace_id: str = "",
        **metadata: Any,
    ) -> Any:
        return self.ctx.tracer.record(
            agent=self.name,
            step=step,
            run_id=run_id,
            trace_id=trace_id,
            input_summary=input_summary,
            output_summary=output_summary,
            rationale=rationale,
            metadata=metadata,
        )

    @asynccontextmanager
    async def timed(self, run_id: UUID | None, step: str, **meta: Any) -> AsyncIterator[None]:
        """Context manager that records latency, metrics and a decision trace."""
        start = time.perf_counter()
        try:
            yield
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            self.ctx.metrics.observe(f"{self.name}_latency_ms", latency_ms)
            self.ctx.metrics.incr(f"{self.name}_steps")
            self.trace(run_id, step, latency_ms=latency_ms, **meta)
