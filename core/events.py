"""Event types carried on the message bus.

Events are the *only* way agents communicate asynchronously, which makes every
decision replayable and lets the same workflow run in-process (tests) or across
processes (production Redis Streams).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from cloud_orchestra.schemas import utcnow


class EventType(StrEnum):
    ALERT_RECEIVED = "alert_received"
    RUN_STARTED = "run_started"
    PROVIDER_SELECTED = "provider_selected"
    MEMORY_RETRIEVED = "memory_retrieved"
    TERRAFORM_GENERATED = "terraform_generated"
    REVIEW_COMPLETED = "review_completed"
    RED_TEAM_COMPLETED = "red_team_completed"
    COST_OPTIMIZED = "cost_optimized"
    PR_OPENED = "pr_opened"
    APPLIED = "applied"
    VERIFIED = "verified"
    ROLLED_BACK = "rolled_back"
    MEMORY_STORED = "memory_stored"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


class Event(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: EventType
    run_id: UUID | None = None
    alert_id: UUID | None = None
    agent: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = ""
    timestamp: datetime = Field(default_factory=utcnow)
