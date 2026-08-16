"""Domain errors and error codes for Cloud-Orchestra.

A compact, typed error hierarchy keeps agent failures distinguishable from
infrastructure failures and lets the orchestrator implement precise retry and
compensation logic.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    CONFIG_ERROR = "CONFIG_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    LLM_ERROR = "LLM_ERROR"
    LLM_PARSE_ERROR = "LLM_PARSE_ERROR"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    TERRAFORM_ERROR = "TERRAFORM_ERROR"
    GITHUB_ERROR = "GITHUB_ERROR"
    SANDBOX_ERROR = "SANDBOX_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    RL_ERROR = "RL_ERROR"
    WORKFLOW_ERROR = "WORKFLOW_ERROR"
    NOT_FOUND = "NOT_FOUND"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"


class CloudOrchestraError(Exception):
    """Base error for the whole project."""

    code: ErrorCode = ErrorCode.WORKFLOW_ERROR

    def __init__(self, message: str, *, code: ErrorCode | None = None, **ctx: object) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.ctx = ctx

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code.value}] {self.message}"


class ConfigError(CloudOrchestraError):
    code = ErrorCode.CONFIG_ERROR


class ValidationError(CloudOrchestraError):
    code = ErrorCode.VALIDATION_ERROR


class LLMError(CloudOrchestraError):
    code = ErrorCode.LLM_ERROR


class LLMParseError(CloudOrchestraError):
    code = ErrorCode.LLM_PARSE_ERROR


class ProviderError(CloudOrchestraError):
    code = ErrorCode.PROVIDER_ERROR


class TerraformError(CloudOrchestraError):
    code = ErrorCode.TERRAFORM_ERROR


class GitHubError(CloudOrchestraError):
    code = ErrorCode.GITHUB_ERROR


class SandboxError(CloudOrchestraError):
    code = ErrorCode.SANDBOX_ERROR


class MemoryError(CloudOrchestraError):
    code = ErrorCode.MEMORY_ERROR


class RLError(CloudOrchestraError):
    code = ErrorCode.RL_ERROR


class NotFoundError(CloudOrchestraError):
    code = ErrorCode.NOT_FOUND


class VerificationFailedError(CloudOrchestraError):
    code = ErrorCode.VERIFICATION_FAILED


class RollbackFailedError(CloudOrchestraError):
    code = ErrorCode.ROLLBACK_FAILED
