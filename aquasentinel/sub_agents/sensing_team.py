"""Stage 1 — 병렬 감시팀 (ParallelAgent, Fan-Out).

세 분석가가 서로 다른 MCP 도구로 동시에 근거를 수집한다. 각 결과 요약은 output_key 로,
도구 원시 결과는 after_tool_callback 이 state['ev:<tool>'] 로 분리 저장한다(병렬 쓰기 충돌 방지).
"""

from google.adk.agents import LlmAgent, ParallelAgent
from google.genai import types

from .. import callbacks, prompts, tools
from ..config import mcp_toolset, model_for

_DET = types.GenerateContentConfig(temperature=0)


def build_sensing_team() -> ParallelAgent:
    water = LlmAgent(
        name="water_quality_analyst",
        model=model_for("water"),
        description="용존산소 시계열로 빈산소 발생 구간과 산소포화도를 분석",
        instruction=prompts.WATER_ANALYST,
        tools=[mcp_toolset(["analyze_hypoxia", "get_sensor_timeseries"]), tools.compute_do_saturation],
        after_tool_callback=callbacks.evidence_ledger_after_tool,
        generate_content_config=_DET,
        output_key="water_diag",
    )
    thermal = LlmAgent(
        name="thermal_stress_analyst",
        model=model_for("thermal"),
        description="일 최고수온·지속일수로 고수온 특보 단계와 열스트레스를 분석",
        instruction=prompts.THERMAL_ANALYST,
        tools=[mcp_toolset(["assess_heat_alert"])],
        after_tool_callback=callbacks.evidence_ledger_after_tool,
        generate_content_config=_DET,
        output_key="thermal_diag",
    )
    weather = LlmAgent(
        name="weather_alert_analyst",
        model=model_for("weather"),
        description="향후 기상(폭염·무풍)과 해역 기관 특보를 분석",
        instruction=prompts.WEATHER_ANALYST,
        tools=[mcp_toolset(["get_weather_forecast", "get_official_alerts"])],
        after_tool_callback=callbacks.evidence_ledger_after_tool,
        generate_content_config=_DET,
        output_key="weather_diag",
    )
    return ParallelAgent(
        name="sensing_team",
        description="수질·고수온·기상 3개 관점을 병렬로 동시에 진단하는 감시팀",
        sub_agents=[water, thermal, weather],
    )
