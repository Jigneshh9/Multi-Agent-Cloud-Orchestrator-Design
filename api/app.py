"""FastAPI control plane."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cloud_orchestra.core.config import Settings, get_settings
from cloud_orchestra.core.metrics import get_registry
from cloud_orchestra.runtime import Runtime
from cloud_orchestra.schemas import Alert, AlertSource


class IngestRequest(BaseModel):
    source: AlertSource
    payload: dict[str, Any] = Field(default_factory=dict)
    trigger_run: bool = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    runtime = Runtime(settings, persistent=True)
    explanations: dict[UUID, str] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await runtime.init_db()
        yield
        await runtime.close()

    app = FastAPI(title="Cloud-Orchestra", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "cloud-orchestra"}

    @app.post("/alerts/ingest")
    async def ingest_alert(request: IngestRequest) -> dict[str, Any]:
        from cloud_orchestra.agents.monitoring import parse_alert

        alert = parse_alert(request.source, request.payload)
        if request.trigger_run:
            result = await runtime.run_alert(alert)
            if result.explanation:
                explanations[result.run.id] = result.explanation
            return {
                "alert": alert.model_dump(mode="json"),
                "run_id": str(result.run.id),
                "status": result.run.status.value,
                "resolved": result.run.resolved,
            }
        return {"alert": alert.model_dump(mode="json")}

    @app.post("/runs")
    async def create_run(alert: Alert) -> dict[str, Any]:
        result = await runtime.run_alert(alert)
        if result.explanation:
            explanations[result.run.id] = result.explanation
        return {
            "run_id": str(result.run.id),
            "status": result.run.status.value,
            "resolved": result.run.resolved,
            "provider": result.run.provider.value if result.run.provider else None,
        }

    @app.get("/runs")
    async def list_runs() -> list[dict[str, Any]]:
        runs = await runtime.list_runs()
        return [
            {
                "run_id": r.id,
                "status": r.status.value,
                "provider": r.provider.value if r.provider else None,
                "resolved": r.resolved,
            }
            for r in runs
        ]

    @app.get("/runs/{run_id}")
    async def get_run(run_id: UUID) -> dict[str, Any]:
        run = await runtime.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {
            "run_id": str(run.id),
            "status": run.status.value,
            "provider": run.provider.value if run.provider else None,
            "resolved": run.resolved,
            "cost_before": run.cost_before,
            "cost_after": run.cost_after,
            "explanation": explanations.get(run_id, ""),
        }

    @app.get("/runs/{run_id}/explanation")
    async def get_explanation(run_id: UUID) -> dict[str, str]:
        explanation = explanations.get(run_id)
        if explanation is None:
            raise HTTPException(status_code=404, detail="explanation not found")
        return {"run_id": str(run_id), "explanation": explanation}

    @app.get("/metrics")
    async def metrics() -> Any:
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(get_registry().to_prometheus_text())

    return app


app = create_app()
