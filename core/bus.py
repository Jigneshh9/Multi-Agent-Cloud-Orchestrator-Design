"""Event bus abstraction.

Two implementations are provided:

* :class:`InMemoryEventBus` — synchronous fan-out for tests and the single-
  process "modular monolith" deployment mode.
* :class:`RedisStreamBus` — Redis Streams with consumer groups for the
  distributed microservices deployment mode.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from cloud_orchestra.core.errors import ProviderError
from cloud_orchestra.core.events import Event, EventType

logger = logging.getLogger("cloud_orchestra.bus")

Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Common interface for event buses."""

    async def start(self) -> None:  # pragma: no cover - trivial
        return None

    async def close(self) -> None:  # pragma: no cover - trivial
        return None

    async def publish(self, event: Event) -> None:
        raise NotImplementedError

    async def subscribe(self, event_type: EventType, handler: Handler) -> None:
        raise NotImplementedError


class InMemoryEventBus(EventBus):
    """Fan-out bus backed by in-process handler registries."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._published: list[Event] = []

    async def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    async def publish(self, event: Event) -> None:
        self._published.append(event)
        handlers = list(self._handlers.get(event.type, []))
        await asyncio.gather(*(handler(event) for handler in handlers))

    @property
    def published(self) -> list[Event]:
        return list(self._published)


class RedisStreamBus(EventBus):
    """Redis Streams implementation.

    Events are appended to a single stream ``cloud-orchestra:events`` and
    distributed to one consumer per group using consumer groups, giving
    at-least-once delivery with per-agent parallelism.
    """

    STREAM = "cloud-orchestra:events"
    GROUP = "cloud-orchestra-agents"

    def __init__(self, redis_url: str, consumer_name: str) -> None:
        self._redis_url = redis_url
        self._consumer_name = consumer_name
        self._client: Any = None
        self._tasks: list[asyncio.Task[None]] = []
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)

    async def start(self) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ProviderError("redis is not installed", code=__import__(
                "cloud_orchestra.core.errors", fromlist=["ErrorCode"]
            ).ErrorCode.PROVIDER_ERROR) from exc
        self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        with contextlib.suppress(Exception):
            await self._client.xgroup_create(self.STREAM, self.GROUP, id="0", mkstream=True)

    async def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    async def publish(self, event: Event) -> None:
        if self._client is None:
            raise ProviderError("bus not started")
        payload = event.model_dump_json()
        await self._client.xadd(self.STREAM, {"event": payload})

    async def consume(self) -> None:  # pragma: no cover - exercised in integration
        """Blocking consume loop; run as a background task in production."""
        assert self._client is not None
        while True:
            messages = await self._client.xreadgroup(
                self.GROUP, self._consumer_name, {self.STREAM: ">"}, count=10, block=5000
            )
            for _stream, entries in messages:
                for _msg_id, fields in entries:
                    raw = fields.get("event", "{}")
                    event = Event.model_validate_json(raw)
                    for handler in self._handlers.get(event.type, []):
                        await handler(event)

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def build_bus(redis_url: str, *, consumer_name: str = "orchestrator", use_redis: bool = False) -> EventBus:
    if use_redis:
        return RedisStreamBus(redis_url, consumer_name)
    return InMemoryEventBus()


def serialize_event(event: Event) -> str:
    return json.dumps(event.model_dump(mode="json"), default=str)
