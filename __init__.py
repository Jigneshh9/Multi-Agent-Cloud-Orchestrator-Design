"""Cloud-Orchestra: a multi-agent cloud DevOps orchestrator.

The package is organised into the following layers:

* ``core``   — configuration, event bus, LLM client, tracing, metrics, errors.
* ``schemas``— Pydantic domain contracts (alerts, terraform plans, findings).
* ``db``     — SQLAlchemy persistence models and repositories.
* ``providers`` — adapters for clouds, GitHub, sandboxes and Terraform.
* ``agents`` — the ten specialised agents (monitoring, devops, review, red-team,
  fin-ops, verifier, rollback, memory-curator, explainer, cloud-harmonizer).
* ``orchestrator`` — the saga workflow that coordinates agents end-to-end.
* ``rl``     — the FinOps reinforcement-learning environment and PPO trainer.
* ``memory`` — the persistent vector memory (RAG) abstraction.
* ``eval``   — the evaluation framework and ablation studies.
* ``api``    — the FastAPI control plane.
"""

__version__ = "0.1.0"

from cloud_orchestra.core.config import Settings, get_settings

__all__ = ["Settings", "__version__", "get_settings"]
