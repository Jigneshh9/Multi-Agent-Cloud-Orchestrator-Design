"""Runtime configuration and feature flags.

Configuration is loaded from environment variables (optionally a ``.env`` file)
so that the same code runs identically in development, tests and containers.
Feature flags are first-class citizens: the evaluation framework toggles them
to run the five ablation studies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class FeatureFlags:
    """Toggle each novelty vector independently (used by ablation studies)."""

    verifier: bool = True  # closed-loop verification
    rollback: bool = True  # automatic revert on failed verification
    red_team: bool = True  # adversarial sandbox penetration testing
    fin_ops_rl: bool = True  # PPO-driven cost optimization
    memory: bool = True  # persistent vector memory (RAG)
    cloud_harmonizer: bool = True  # multi-cloud provider selection
    explainer: bool = True  # human-readable decision explanations

    def disable(self, name: str) -> FeatureFlags:
        """Return a copy with a single flag disabled (for ablation runs)."""
        if not hasattr(self, name):
            raise ValueError(f"unknown feature flag: {name}")
        return replace(self, **{name: False})


@dataclass(frozen=True)
class Settings:
    env: str = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite+aiosqlite:///cloud_orchestra.db"
    redis_url: str = "redis://localhost:6379/0"

    # LLM
    llm_provider: str = "openai_compatible"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-v4"
    llm_temperature: float = 0.1
    llm_timeout_seconds: float = 120.0

    # GitHub
    github_token: str = ""
    github_repo_owner: str = ""
    github_repo_name: str = ""
    github_base_branch: str = "main"

    # Clouds
    aws_region: str = "us-east-1"
    gcp_project_id: str = ""
    azure_subscription_id: str = ""

    # Sandbox
    sandbox_provider: str = "mock"

    # Memory
    memory_provider: str = "memory"
    chroma_persist_dir: str = "./chroma_data"

    # FinOps RL
    fin_ops_rl_model_path: str = "./models/finops_ppo.pt"

    # Metrics / tracing
    otel_enabled: bool = False
    service_name: str = "cloud-orchestra"

    features: FeatureFlags = field(default_factory=FeatureFlags)

    @classmethod
    def from_env(cls) -> Settings:
        try:  # pragma: no cover - dotenv is optional nicety
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass

        return cls(
            env=_env("CLOUD_ORCHESTRA_ENV", "development") or "development",
            log_level=_env("CLOUD_ORCHESTRA_LOG_LEVEL", "INFO") or "INFO",
            database_url=_env(
                "CLOUD_ORCHESTRA_DATABASE_URL", "sqlite+aiosqlite:///cloud_orchestra.db"
            )
            or "sqlite+aiosqlite:///cloud_orchestra.db",
            redis_url=_env("CLOUD_ORCHESTRA_REDIS_URL", "redis://localhost:6379/0")
            or "redis://localhost:6379/0",
            llm_provider=_env("LLM_PROVIDER", "openai_compatible") or "openai_compatible",
            llm_base_url=_env("LLM_BASE_URL", "https://api.deepseek.com/v1")
            or "https://api.deepseek.com/v1",
            llm_api_key=_env("LLM_API_KEY", "") or "",
            llm_model=_env("LLM_MODEL", "deepseek-v4") or "deepseek-v4",
            llm_temperature=_float(_env("LLM_TEMPERATURE"), 0.1),
            llm_timeout_seconds=_float(_env("LLM_TIMEOUT_SECONDS"), 120.0),
            github_token=_env("GITHUB_TOKEN", "") or "",
            github_repo_owner=_env("GITHUB_REPO_OWNER", "") or "",
            github_repo_name=_env("GITHUB_REPO_NAME", "") or "",
            github_base_branch=_env("GITHUB_BASE_BRANCH", "main") or "main",
            aws_region=_env("AWS_REGION", "us-east-1") or "us-east-1",
            gcp_project_id=_env("GCP_PROJECT_ID", "") or "",
            azure_subscription_id=_env("AZURE_SUBSCRIPTION_ID", "") or "",
            sandbox_provider=_env("SANDBOX_PROVIDER", "mock") or "mock",
            memory_provider=_env("MEMORY_PROVIDER", "memory") or "memory",
            chroma_persist_dir=_env("CHROMA_PERSIST_DIR", "./chroma_data") or "./chroma_data",
            fin_ops_rl_model_path=_env("FIN_OPS_RL_MODEL_PATH", "./models/finops_ppo.pt")
            or "./models/finops_ppo.pt",
            otel_enabled=_bool(_env("OTEL_ENABLED"), False),
            service_name=_env("OTEL_SERVICE_NAME", "cloud-orchestra") or "cloud-orchestra",
            features=FeatureFlags(
                verifier=_bool(_env("FEATURE_VERIFIER"), True),
                rollback=_bool(_env("FEATURE_ROLLBACK"), True),
                red_team=_bool(_env("FEATURE_RED_TEAM"), True),
                fin_ops_rl=_bool(_env("FEATURE_FIN_OPS_RL"), True),
                memory=_bool(_env("FEATURE_MEMORY"), True),
                cloud_harmonizer=_bool(_env("FEATURE_CLOUD_HARMONIZER"), True),
                explainer=_bool(_env("FEATURE_EXPLAINER"), True),
            ),
        )

    def with_features(self, features: FeatureFlags) -> Settings:
        return replace(self, features=features)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def get_settings_uncached() -> Settings:
    """Return a fresh settings object (used by tests and ablation runs)."""
    return Settings.from_env()
