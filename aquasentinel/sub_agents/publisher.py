"""Stage 4 — 보고서 발행 (차트 Artifact + Markdown Artifact)."""

from google.adk.agents import LlmAgent
from google.genai import types

from .. import callbacks, prompts, tools
from ..config import model_for


def build_report_publisher() -> LlmAgent:
    return LlmAgent(
        name="report_publisher",
        model=model_for("publisher"),
        description="검증 통과 초안을 추세 차트와 함께 보고서 Artifact 로 발행",
        instruction=prompts.REPORT_PUBLISHER,
        tools=[tools.render_trend_chart, tools.publish_report],
        before_tool_callback=callbacks.safety_interlock_before_tool,
        generate_content_config=types.GenerateContentConfig(temperature=0),
        output_key="final_summary",
    )
