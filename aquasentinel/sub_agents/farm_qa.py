"""일반 질의 응답 에이전트 (Coordinator 의 동적 위임 대상)."""

from google.adk.agents import LlmAgent

from .. import callbacks, prompts, tools
from ..config import mcp_toolset, model_for


def build_farm_qa_agent() -> LlmAgent:
    return LlmAgent(
        name="farm_qa_agent",
        model=model_for("farm_qa"),
        description="양식장 목록·대장, 해역 특보, 특보 기준·관리요령 등 일반 질문에 답변",
        instruction=prompts.FARM_QA,
        tools=[mcp_toolset(["list_farms", "get_farm_profile", "get_official_alerts"]),
               tools.lookup_response_manual],
        before_model_callback=callbacks.prompt_guard_before_model,
    )
