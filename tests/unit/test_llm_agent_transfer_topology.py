"""Specialists must be able to hand a conversation to their peers or parent."""

from types import SimpleNamespace

import pytest
from google.adk.agents.llm_agent import LlmAgent
from google.adk.flows.llm_flows.agent_transfer import _get_transfer_targets

from src.services.adk.agents.llm_agent_builder import LlmAgentBuilder


@pytest.mark.asyncio
async def test_specialist_can_transfer_to_sales_after_reception_handoff(monkeypatch):
    questions = LlmAgent(name="bee_duvidas")
    sales = LlmAgent(name="bee_comercial")
    reception = LlmAgent(name="bee_recepcao")

    async def load_children(*args, **kwargs):
        return [questions, sales], []

    async def create_agent(*args, **kwargs):
        return reception, []

    monkeypatch.setattr(
        "src.services.adk.agents.llm_agent_builder.get_sub_agents", load_children
    )
    monkeypatch.setattr(LlmAgentBuilder, "_create_llm_agent", create_agent)
    config = SimpleNamespace(type="llm", config={"sub_agents": ["questions", "sales"]})

    root, _ = await LlmAgentBuilder(None).build_llm_agent(config)

    assert [agent.name for agent in _get_transfer_targets(root)] == [
        "bee_duvidas", "bee_comercial"
    ]
    assert [agent.name for agent in _get_transfer_targets(questions)] == [
        "bee_recepcao", "bee_comercial"
    ]
    assert questions.root_agent is root
    assert sales.root_agent is root
