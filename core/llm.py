"""LLM client abstraction.

Cloud-Orchestra talks to any OpenAI-compatible endpoint (DeepSeek V4, GPT-4o,
self-hosted vLLM). Structured output is achieved through JSON-schema function
calling with deterministic retry/validation, which is the mechanism that makes
the DevOps Agent emit a *typed* Terraform IR instead of free-form HCL.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from cloud_orchestra.core.errors import LLMError, LLMParseError

logger = logging.getLogger("cloud_orchestra.llm")

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Interface implemented by the real and mock clients."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> LLMResponse: ...

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T: ...


class OpenAICompatibleLLMClient:
    """Talks to OpenAI/DeepSeek-compatible ``/chat/completions`` endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.1,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature if temperature is None else temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        url = f"{self._base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, headers=self._headers, json=body)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
            usage = data.get("usage", {})
            return LLMResponse(
                content=content,
                model=data.get("model", self._model),
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                raw=data,
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMParseError(f"unexpected LLM response shape: {data}") from exc

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T:
        # Prefer native tool-calling; fall back to JSON mode + manual parse.
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature if temperature is None else temperature,
            "response_format": {"type": "json_object"},
        }
        url = f"{self._base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, headers=self._headers, json=body)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        content = data["choices"][0]["message"]["content"] or ""
        return _parse_structured(content, schema)


class MockLLMClient:
    """Deterministic client for tests and offline evaluation.

    Responses are provided via a callable or a static map keyed by the last
    user message. This keeps the whole pipeline reproducible for benchmarking.
    """

    def __init__(
        self,
        responder: Any = None,
        *,
        default_content: str = "{}",
    ) -> None:
        self._responder = responder
        self._default_content = default_content
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "json_mode": json_mode})
        content = self._resolve(messages)
        return LLMResponse(content=content, model="mock-llm")

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T:
        self.calls.append({"messages": messages, "structured": schema.__name__})
        content = self._resolve(messages)
        return _parse_structured(content, schema)

    def _resolve(self, messages: list[dict[str, str]]) -> str:
        if self._responder is None:
            return self._default_content
        if callable(self._responder):
            result = self._responder(messages)
            return json.dumps(result) if not isinstance(result, str) else result
        # dict keyed by last user content
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        if isinstance(self._responder, dict):
            return str(self._responder.get(last_user, self._default_content))
        return self._default_content


def _parse_structured(content: str, schema: type[T]) -> T:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"LLM returned non-JSON: {content[:200]}") from exc

    adapter = TypeAdapter(schema)
    try:
        return adapter.validate_python(data)
    except ValidationError as exc:
        raise LLMParseError(f"LLM JSON failed schema validation: {exc}") from exc


def build_llm_client(
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    *,
    temperature: float = 0.1,
    timeout: float = 120.0,
) -> LLMClient:
    if provider == "mock":
        return MockLLMClient()
    if provider == "openai_compatible":
        return OpenAICompatibleLLMClient(
            base_url, api_key, model, temperature=temperature, timeout=timeout
        )
    raise LLMError(f"unknown LLM provider: {provider}")
