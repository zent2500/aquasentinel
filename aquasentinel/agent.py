"""AquaSentinel — 해상 양식장 고수온·빈산소 위기대응 멀티에이전트 (ADK 진입점).

에이전트 트리
aqua_coordinator (LlmAgent, 동적 위임)
├── farm_qa_agent (LlmAgent + MCP + RAG)                 ← 일반 질의
└── crisis_pipeline (SequentialAgent)                    ← 위기 진단
    ├── sensing_team (ParallelAgent)
    │   ├── water_quality_analyst   (MCP + 과학계산 도구)
    │   ├── thermal_stress_analyst  (MCP)
    │   └── weather_alert_analyst   (MCP)
    ├── risk_gate (RiskGateAgent: BaseAgent, LLM 미사용 결정론 판정)
    ├── response_loop (LoopAgent, max 3)
    │   ├── action_planner   (RAG + 장비제어 · 안전 인터록)
    │   └── grounding_critic (결정론 검증 도구 + exit_loop)
    └── report_publisher (차트·보고서 Artifact)
"""

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.apps import App

from . import callbacks, prompts, tools
from .config import model_for
from .plugins import AuditTrailPlugin
from .risk_gate import RiskGateAgent
from .sub_agents import (build_farm_qa_agent, build_report_publisher, build_response_loop,
                         build_sensing_team)

crisis_pipeline = SequentialAgent(
    name="crisis_pipeline",
    description="병렬 감시 → 결정론 위험판정 → 작성·검증 루프 → 보고서 발행 순차 파이프라인",
    sub_agents=[
        build_sensing_team(),
        RiskGateAgent(name="risk_gate", description="근거 원장 기반 ARI 위기지수 결정론 판정"),
        build_response_loop(),
        build_report_publisher(),
    ],
)

root_agent = LlmAgent(
    name="aqua_coordinator",
    model=model_for("coordinator"),
    description="양식장 위기대응 요청을 접수하고 위기 파이프라인 또는 Q&A 에이전트로 동적 위임",
    instruction=prompts.COORDINATOR,
    tools=[tools.register_incident],
    before_model_callback=callbacks.prompt_guard_before_model,
    sub_agents=[crisis_pipeline, build_farm_qa_agent()],
)

app = App(name="aquasentinel", root_agent=root_agent, plugins=[AuditTrailPlugin()])
