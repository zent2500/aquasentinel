"""Stage 3 — 작성·검증 루프 (LoopAgent, Generator-Critic).

Critic 은 LLM 의 '느낌'이 아니라 결정론적 도구 verify_numeric_grounding 의 결과로만 판정한다.
max_iterations=3 + exit_loop(escalate) 로 무한 루프를 방지한다.
"""

from google.adk.agents import LlmAgent, LoopAgent
from google.genai import types

from .. import callbacks, prompts, tools
from ..config import model_for

MAX_ITERATIONS = 3


def build_response_loop() -> LoopAgent:
    planner = LlmAgent(
        name="action_planner",
        model=model_for("planner"),
        description="RiskGate 판정과 매뉴얼 RAG 근거로 대응계획 초안을 작성·수정",
        instruction=prompts.ACTION_PLANNER,
        tools=[tools.lookup_response_manual, tools.request_equipment_action],
        before_tool_callback=callbacks.safety_interlock_before_tool,
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
        output_key="action_plan_draft",
    )
    critic = LlmAgent(
        name="grounding_critic",
        model=model_for("critic"),
        description="초안의 수치·인용·섹션을 근거 원장과 대조하고 통과 시 루프 종료",
        instruction=prompts.GROUNDING_CRITIC,
        tools=[tools.verify_numeric_grounding, tools.exit_loop],
        generate_content_config=types.GenerateContentConfig(temperature=0),
        output_key="critic_feedback",
    )
    return LoopAgent(
        name="response_loop",
        description="대응계획 작성 → 그라운딩 검증을 최대 3회 반복",
        sub_agents=[planner, critic],
        max_iterations=MAX_ITERATIONS,
    )
