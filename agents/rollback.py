"""Rollback Agent — reverts a failed remediation (novelty #1: self-healing)."""

from __future__ import annotations

from cloud_orchestra.agents.base import BaseAgent
from cloud_orchestra.providers.terraform import TerraformProvider, render_hcl
from cloud_orchestra.schemas import RollbackResult, TerraformPlan


class RollbackAgent(BaseAgent):
    name = "rollback"

    async def rollback(
        self, plan: TerraformPlan, terraform: TerraformProvider
    ) -> RollbackResult:
        async with self.timed(None, "rollback", input_summary=plan.description):
            hcl = render_hcl(plan)
            result = await terraform.destroy(hcl)
            return RollbackResult(
                succeeded=result.succeeded,
                reverted_resources=result.applied_resources,
                error=result.error,
            )
