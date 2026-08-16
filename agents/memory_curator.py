"""Memory Curator — persistent shared memory with evolution (novelty #4).

Stores each deployment's outcome in the vector store so future runs can
retrieve similar past experiences (RAG) and get progressively smarter.
"""

from __future__ import annotations

from cloud_orchestra.agents.base import BaseAgent
from cloud_orchestra.schemas import Alert, MemoryEntry, Run, TerraformPlan


class MemoryCurator(BaseAgent):
    name = "memory_curator"

    async def store(
        self,
        run: Run,
        alert: Alert,
        plan: TerraformPlan,
        *,
        resolved: bool,
        cost_delta_usd: float,
    ) -> MemoryEntry:
        summary = (
            f"{alert.problem_class} on {alert.resource_type} ({plan.provider.value}) "
            f"was {'resolved' if resolved else 'not resolved'} with cost delta "
            f"${cost_delta_usd:.2f}."
        )
        entry = MemoryEntry(
            run_id=run.id,
            problem_class=alert.problem_class,
            provider=plan.provider,
            resource_type=alert.resource_type,
            summary=summary,
            terraform_plan=plan,
            success=True,
            cost_delta_usd=cost_delta_usd,
            resolved=resolved,
            latency_ms=0,
        )
        async with self.timed(run.id, "store_memory", input_summary=summary):
            await self.ctx.repository.save_memory(entry)
            await self.ctx.memory.add(entry, summary)
        return entry

    async def retrieve(self, alert: Alert, top_k: int = 5) -> list[MemoryEntry]:
        query = f"{alert.problem_class} {alert.resource_type}"
        async with self.timed(None, "retrieve_memory", input_summary=query):
            hits = await self.ctx.memory.search(query, top_k=top_k)
        return [entry for entry, _score in hits]
