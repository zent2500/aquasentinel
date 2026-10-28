"""RiskGateAgent — LLM을 쓰지 않는 결정론적 커스텀 에이전트(BaseAgent 상속).

병렬 분석팀이 남긴 근거 원장(state['ev:*'])만으로 ARI 위기지수와 단계를 계산한다.
'점수는 코드가, 설명은 LLM이' 원칙: 위험 단계 판정에 LLM 의 주관이 개입하지 않는다.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from . import science


class RiskGateAgent(BaseAgent):
    """근거 원장을 읽어 ARI(0~100)와 위험 단계를 산출하고 state['risk_assessment']에 기록."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        st = ctx.session.state
        result = science.compute_ari(st.get("ev:analyze_hypoxia"), st.get("ev:assess_heat_alert"),
                                     st.get("ev:get_weather_forecast"))
        sat = st.get("ev:compute_do_saturation")
        if sat and result.get("status") == "success":
            result["basis"]["do_saturation_percent"] = sat.get("do_saturation_percent")
        if result["status"] == "success":
            c = result["components"]
            msg = (f"🧮 RiskGate(결정론 판정) — ARI {result['ari_score']}점 → **{result['risk_level']}** 단계\n"
                   f"- 고수온 {c['heat_points']} + 최저DO {c['do_min_points']} + 빈산소지속 "
                   f"{c['hypoxia_duration_points']} + 기상 {c['weather_points']}\n"
                   f"- 단계 구간: {result['level_cutoffs']}")
        else:
            msg = f"🧮 RiskGate — 근거 부족으로 판정 불가: {result['missing_evidence']}"
        print(f"[RiskGate] {json.dumps(result, ensure_ascii=False)[:200]}")
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(text=msg)]),
            actions=EventActions(state_delta={"risk_assessment": result,
                                              "risk_level": result["risk_level"]}),
        )
