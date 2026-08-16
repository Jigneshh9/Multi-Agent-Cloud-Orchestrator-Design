"""Runtime — wires configuration into a runnable orchestrator.

A :class:`Runtime` owns long-lived, stateless services (LLM client, tracer,
metrics, memory store, providers, event bus, RL policy) and creates a fresh
database session + repository for each workflow run. Passing ``persistent=False``
uses the in-memory repository (used by the evaluation harness and tests).
"""

from __future__ import annotations

from typing import Any

from cloud_orchestra.agents.base import AgentContext
from cloud_orchestra.agents.registry import AgentRegistry, build_agents
from cloud_orchestra.core.bus import EventBus, build_bus
from cloud_orchestra.core.config import Settings
from cloud_orchestra.core.llm import LLMClient, build_llm_client
from cloud_orchestra.core.metrics import MetricsRegistry, get_registry
from cloud_orchestra.core.tracing import Tracer
from cloud_orchestra.db.repository import InMemoryRepository, Repository
from cloud_orchestra.db.session import create_engine, create_sessionmaker
from cloud_orchestra.memory.store import MemoryStore, build_memory_store
from cloud_orchestra.orchestrator.workflow import Orchestrator, WorkflowResult
from cloud_orchestra.providers.cloud import CloudClient, build_cloud_client
from cloud_orchestra.providers.github import GitHubClient, build_github_client
from cloud_orchestra.providers.sandbox import SandboxProvider, build_sandbox_provider
from cloud_orchestra.providers.terraform import TerraformProvider, build_terraform_provider
from cloud_orchestra.schemas import Alert


class Runtime:
    def __init__(self, settings: Settings, *, persistent: bool = True) -> None:
        self.settings = settings

        self.llm: LLMClient = build_llm_client(
            settings.llm_provider,
            settings.llm_base_url,
            settings.llm_api_key,
            settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
        self.tracer = Tracer()
        self.metrics: MetricsRegistry = get_registry()
        self.memory: MemoryStore = build_memory_store(
            settings.memory_provider, settings.chroma_persist_dir
        )
        self.cloud: CloudClient = build_cloud_client()
        self.github: GitHubClient = build_github_client(settings.github_token)
        self.sandbox: SandboxProvider = build_sandbox_provider(settings.sandbox_provider)
        self.terraform: TerraformProvider = build_terraform_provider()
        self.bus: EventBus = build_bus(settings.redis_url)
        self.finops_policy: Any = self._build_finops_policy(settings)

        self._persistent = persistent
        self._engine = None
        self._sessionmaker = None
        if persistent:
            self._engine = create_engine(settings.database_url)
            self._sessionmaker = create_sessionmaker(self._engine)

    def _build_finops_policy(self, settings: Settings) -> Any:
        if not settings.features.fin_ops_rl:
            return None
        from cloud_orchestra.rl.policy import FinOpsPolicy

        return FinOpsPolicy(settings.fin_ops_rl_model_path)

    def make_context(self, repository: Repository | InMemoryRepository) -> AgentContext:
        return AgentContext(
            settings=self.settings,
            llm=self.llm,
            tracer=self.tracer,
            metrics=self.metrics,
            repository=repository,
            memory=self.memory,
            cloud=self.cloud,
            github=self.github,
            sandbox=self.sandbox,
            terraform=self.terraform,
            bus=self.bus,
            finops_policy=self.finops_policy,
        )

    async def run_alert(self, alert: Alert) -> WorkflowResult:
        if self._sessionmaker is None:
            repository: Repository | InMemoryRepository = InMemoryRepository()
            return await self._run_with(repository, alert)

        async with self._sessionmaker() as session:
            repository = Repository(session)
            result = await self._run_with(repository, alert)
            await repository.commit()
            return result

    async def _run_with(
        self, repository: Repository | InMemoryRepository, alert: Alert
    ) -> WorkflowResult:
        ctx = self.make_context(repository)
        agents: AgentRegistry = build_agents(ctx)
        orchestrator = Orchestrator(ctx, agents)
        return await orchestrator.run(alert)

    async def close(self) -> None:
        await self.memory.close()
        await self.bus.close()
        if self._engine is not None:
            await self._engine.dispose()

    async def init_db(self) -> None:
        if self._engine is not None:
            from cloud_orchestra.db.session import init_db

            await init_db(self._engine)

    async def list_runs(self, status: Any = None) -> list[Any]:
        if self._sessionmaker is None:
            return []
        async with self._sessionmaker() as session:
            repo = Repository(session)
            return await repo.list_runs(status)

    async def get_run(self, run_id: Any) -> Any:
        if self._sessionmaker is None:
            return None
        async with self._sessionmaker() as session:
            repo = Repository(session)
            return await repo.get_run(run_id)
