"""Terraform intermediate-representation renderer and lifecycle adapters.

The DevOps Agent emits a typed :class:`TerraformPlan`; :func:`render_hcl`
deterministically serialises it to HCL. Lifecycle adapters then run
``validate``/``plan``/``apply``/``destroy`` either against the real ``terraform``
CLI (:class:`LocalTerraformProvider`) or in dry-run mode
(:class:`DryRunTerraformProvider`) for tests and offline evaluation.
"""

from __future__ import annotations

import asyncio
from typing import Any

from cloud_orchestra.core.errors import TerraformError
from cloud_orchestra.schemas import ApplyResult, TerraformPlan


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        items = [f"{k} = {_literal(v)}" for k, v in value.items()]
        return "{ " + ", ".join(items) + " }"
    return _literal(str(value))


def render_hcl(plan: TerraformPlan) -> str:
    lines: list[str] = []
    lines.append(f'provider "{plan.provider.value}" {{')
    for key, value in plan.provider_config.items():
        lines.append(f"  {key} = {_literal(value)}")
    lines.append("}")
    lines.append("")

    for name, value in plan.variables.items():
        lines.append(f'variable "{name}" {{')
        lines.append(f"  default = {_literal(value)}")
        lines.append("}")
        lines.append("")

    for resource in plan.resources:
        depends = ""
        if resource.depends_on:
            depends = ", ".join(resource.depends_on)
            depends = f", depends_on = [{depends}]"
        lines.append(f'resource "{resource.resource_type}" "{resource.name}" {{')
        for key, value in resource.attributes.items():
            lines.append(f"  {key} = {_literal(value)}")
        lines.append("}")
        lines.append("")

    for name, value in plan.outputs.items():
        lines.append(f'output "{name}" {{')
        lines.append(f"  value = {_literal(value)}")
        lines.append("}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


class TerraformProvider:
    async def validate(self, hcl: str) -> bool:
        raise NotImplementedError

    async def apply(self, hcl: str, *, var_file: str | None = None) -> ApplyResult:
        raise NotImplementedError

    async def destroy(self, hcl: str) -> ApplyResult:
        raise NotImplementedError


class DryRunTerraformProvider(TerraformProvider):
    """Deterministic provider that never touches a real cloud or the CLI.

    It simulates an apply by extracting the resource addresses from the HCL and
    returns success. This is the default for tests and the evaluation harness.
    """

    async def validate(self, hcl: str) -> bool:
        if 'resource "' not in hcl and "provider" not in hcl:
            raise TerraformError("plan contains no resources or provider")
        return True

    async def apply(self, hcl: str, *, var_file: str | None = None) -> ApplyResult:
        resources = _extract_resource_addresses(hcl)
        await asyncio.sleep(0)
        return ApplyResult(succeeded=True, applied_resources=resources, apply_output="dry-run apply")

    async def destroy(self, hcl: str) -> ApplyResult:
        resources = _extract_resource_addresses(hcl)
        await asyncio.sleep(0)
        return ApplyResult(succeeded=True, applied_resources=resources, apply_output="dry-run destroy")


class LocalTerraformProvider(TerraformProvider):
    """Runs the real ``terraform`` CLI via subprocess (production mode)."""

    def __init__(self, work_dir: str = ".terraform-work") -> None:
        self._work_dir = work_dir

    async def validate(self, hcl: str) -> bool:
        result = await self._run("validate")
        return result.returncode == 0

    async def apply(self, hcl: str, *, var_file: str | None = None) -> ApplyResult:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "main.tf")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(hcl)
            result = await self._run("apply", cwd=tmp, auto_approve=True)
            return ApplyResult(
                succeeded=result.returncode == 0,
                apply_output=result.stdout + result.stderr,
                applied_resources=_extract_resource_addresses(hcl),
                error="" if result.returncode == 0 else result.stderr,
            )

    async def destroy(self, hcl: str) -> ApplyResult:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "main.tf")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(hcl)
            result = await self._run("destroy", cwd=tmp, auto_approve=True)
            return ApplyResult(
                succeeded=result.returncode == 0,
                apply_output=result.stdout + result.stderr,
                applied_resources=_extract_resource_addresses(hcl),
                error="" if result.returncode == 0 else result.stderr,
            )

    async def _run(self, *args: str, cwd: str | None = None, auto_approve: bool = False) -> _CmdResult:
        cmd = ["terraform", *args]
        if auto_approve:
            cmd.append("-auto-approve")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            return _CmdResult(proc.returncode or 1, stdout.decode(), stderr.decode())
        except FileNotFoundError as exc:
            raise TerraformError("terraform CLI not found on PATH") from exc


def _extract_resource_addresses(hcl: str) -> list[str]:
    addresses: list[str] = []
    for line in hcl.splitlines():
        stripped = line.strip()
        if stripped.startswith('resource "'):
            parts = stripped.split('"')
            if len(parts) >= 5:
                addresses.append(f"{parts[1]}.{parts[3]}")
    return addresses


class _CmdResult:
    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def build_terraform_provider(*, local: bool = False, work_dir: str = ".terraform-work") -> TerraformProvider:
    if local:
        return LocalTerraformProvider(work_dir)
    return DryRunTerraformProvider()
