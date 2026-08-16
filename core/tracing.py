"""Decision tracing — the backbone of observability and explainability.

Every agent records a :class:`DecisionTrace` that captures *what* it decided,
*why*, with which model, and how long it took. The Explainer Agent later turns
a run's trace chain into a human-readable narrative, and the evaluation
framework replays traces to compute attribution.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from cloud_orchestra.schemas import utcnow

logger = logging.getLogger("cloud_orchestra.tracing")

_current_run_id: contextvars.ContextVar[UUID | None] = contextvars.ContextVar(
    "cloud_orchestra_run_id", default=None
)
_current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "cloud_orchestra_trace_id", default=""
)


class DecisionTrace(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID | None = None
    trace_id: str = ""
    agent: str
    step: str
    parent_id: UUID | None = None
    input_summary: str = ""
    output_summary: str = ""
    rationale: str = ""
    llm_model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utcnow)


class Tracer:
    """Collects decision traces; can be backed by the repository for persistence.

    A trace ``id`` groups all traces of a single workflow run, and ``parent_id``
    links an agent's sub-decisions (e.g. each Red-Team attack module) to the
    agent-level trace.
    """

    def __init__(self) -> None:
        self._traces: list[DecisionTrace] = []

    @contextmanager
    def context(self, run_id: UUID, trace_id: str) -> Iterator[None]:
        """Set the ambient run context for the duration of a workflow run."""
        rid_token = _current_run_id.set(run_id)
        tid_token = _current_trace_id.set(trace_id)
        try:
            yield
        finally:
            _current_run_id.reset(rid_token)
            _current_trace_id.reset(tid_token)

    def record(
        self,
        *,
        agent: str,
        step: str,
        run_id: UUID | None = None,
        trace_id: str = "",
        parent_id: UUID | None = None,
        input_summary: str = "",
        output_summary: str = "",
        rationale: str = "",
        llm_model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> DecisionTrace:
        if run_id is None:
            run_id = _current_run_id.get()
        if not trace_id:
            trace_id = _current_trace_id.get()
        trace = DecisionTrace(
            run_id=run_id,
            trace_id=trace_id,
            agent=agent,
            step=step,
            parent_id=parent_id,
            input_summary=input_summary,
            output_summary=output_summary,
            rationale=rationale,
            llm_model=llm_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self._traces.append(trace)
        logger.info("trace agent=%s step=%s run=%s", agent, step, run_id)
        return trace

    def for_run(self, run_id: UUID) -> list[DecisionTrace]:
        return [t for t in self._traces if t.run_id == run_id]

    def all(self) -> list[DecisionTrace]:
        return list(self._traces)

    def clear(self) -> None:
        self._traces.clear()
