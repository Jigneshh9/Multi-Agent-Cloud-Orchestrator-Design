"""Agent registry — constructs every agent from a shared runtime context."""

from __future__ import annotations

from dataclasses import dataclass

from cloud_orchestra.agents.base import AgentContext
from cloud_orchestra.agents.cloud_harmonizer import CloudHarmonizer
from cloud_orchestra.agents.devops import DevOpsAgent
from cloud_orchestra.agents.explainer import ExplainerAgent
from cloud_orchestra.agents.finops import FinOpsAgent
from cloud_orchestra.agents.memory_curator import MemoryCurator
from cloud_orchestra.agents.monitoring import MonitoringAgent
from cloud_orchestra.agents.redteam import RedTeamAgent
from cloud_orchestra.agents.review import ReviewAgent
from cloud_orchestra.agents.rollback import RollbackAgent
from cloud_orchestra.agents.verifier import VerifierAgent


@dataclass
class AgentRegistry:
    monitoring: MonitoringAgent
    harmonizer: CloudHarmonizer
    devops: DevOpsAgent
    review: ReviewAgent
    red_team: RedTeamAgent
    fin_ops: FinOpsAgent
    verifier: VerifierAgent
    rollback: RollbackAgent
    memory: MemoryCurator
    explainer: ExplainerAgent


def build_agents(ctx: AgentContext) -> AgentRegistry:
    return AgentRegistry(
        monitoring=MonitoringAgent(ctx),
        harmonizer=CloudHarmonizer(ctx),
        devops=DevOpsAgent(ctx),
        review=ReviewAgent(ctx),
        red_team=RedTeamAgent(ctx),
        fin_ops=FinOpsAgent(ctx),
        verifier=VerifierAgent(ctx),
        rollback=RollbackAgent(ctx),
        memory=MemoryCurator(ctx),
        explainer=ExplainerAgent(ctx),
    )
