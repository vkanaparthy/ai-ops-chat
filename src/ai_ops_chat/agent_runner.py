from __future__ import annotations

import logging

from strands import Agent
from strands.models.bedrock import BedrockModel

from ai_ops_chat.chroma_manager import ChromaManager
from ai_ops_chat.config import Settings
from ai_ops_chat.tools import build_tools

logger = logging.getLogger(__name__)


def _system_prompt(agent_name: str) -> str:
    return f"""Your name is {agent_name}. You are an DevOps AI assistant designed to interact with a Chroma database to perform Root Cause Analysis.

Instructions:
- Call list_logs ONLY ONCE if the user asks what data is available or what logs are indexed.
- Use search_logs for all specific queries regarding errors, failures, symptoms, or root causes.
- When reporting findings, include the full text of all matching log documents from tool results in your reasoning, then summarize.
- Your final answer to the user must be a concise summary formatted in HTML only (use tags such as <h3>, <p>, <ul>, <li>, and <pre> for log excerpts). Do not wrap the entire answer in markdown code fences."""


def run_rca_agent(user_query: str, chroma: ChromaManager, settings: Settings) -> str:
    model = BedrockModel(
        region_name=settings.aws_region,
        model_id=settings.bedrock_model_id,
        streaming=False,
    )
    tools = build_tools(chroma, default_top_k=settings.search_default_top_k)
    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=_system_prompt(settings.agent_name),
    )
    logger.info("agent invoke (query chars=%s)", len(user_query))
    result = agent(user_query)
    return str(result).strip()
