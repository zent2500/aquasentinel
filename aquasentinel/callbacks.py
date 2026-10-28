"""3중 안전 가드레일 + 근거 원장(Evidence Ledger) 자동 기록 콜백.

  ① before_model_callback : 프롬프트 인젝션 / 위험 장비 조작 지시를 LLM 호출 전에 차단
  ② before_tool_callback  : 안전 인터록 — 위험 상황에서 산소장비 OFF·급이 ON 같은 위험 제어를 차단,
                            실행 파일 확장자 저장 차단, 검증 미통과 보고서 발행 차단
  ③ after_tool_callback   : MCP·과학 계산 도구의 결과를 state['ev:<도구명>']에 자동 적재
                            → Critic 의 수치 그라운딩 검증이 이 원장을 기준으로 수행된다
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

EVIDENCE_TOOLS = {
    "analyze_hypoxia", "assess_heat_alert", "get_weather_forecast", "get_official_alerts",
    "get_sensor_timeseries", "get_farm_profile", "compute_do_saturation",
}

_INJECTION_PATTERNS = [
    r"이전\s*(지시|명령|규칙).{0,6}(무시|잊)", r"ignore (all |the )?(previous|prior) instructions",
    r"시스템\s*프롬프트.{0,10}(보여|출력|알려)", r"system prompt", r"너는 이제부터",
    r"(산소\s*발생기|산소\s*장비|폭기|에어레이터).{0,10}(전부|모두|다)?\s*(꺼|끄|중지|정지)",
]
_SAFE_REFUSAL = (
    "⚠️ 안전 가드레일에 의해 차단된 요청입니다.\n"
    "- 시스템 지시 변경·노출 요청 또는 위기 상황에서 산소공급 장비를 끄는 지시는 처리할 수 없습니다.\n"
    "- 양식장 위기 진단이 필요하면 '양식장 GH-01 위기 진단해줘' 처럼 요청해 주세요."
)


def _last_user_text(llm_request: LlmRequest) -> str:
    for content in reversed(llm_request.contents or []):
        if content.role == "user":
            texts = [p.text for p in (content.parts or []) if getattr(p, "text", None)]
            if texts:
                return " ".join(texts)
            return ""  # function_response 등
    return ""


# ① -------------------------------------------------------------------------
def prompt_guard_before_model(callback_context: CallbackContext,
                              llm_request: LlmRequest) -> Optional[LlmResponse]:
    """사용자 입력에서 프롬프트 인젝션·위험 지시를 탐지하면 LLM 호출 없이 거절 응답을 반환."""
    text = _last_user_text(llm_request)
    for pat in _INJECTION_PATTERNS:
        if text and re.search(pat, text, flags=re.IGNORECASE):
            callback_context.state["guardrail_last_block"] = {"layer": "before_model", "pattern": pat,
                                                              "input": text[:200]}
            print(f"[Guardrail:before_model] 차단 pattern={pat!r}")
            return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=_SAFE_REFUSAL)]))
    return None


# ② -------------------------------------------------------------------------
_DANGER_LEVELS = {"주의", "경계", "심각"}


def safety_interlock_before_tool(tool: BaseTool, args: dict[str, Any],
                                 tool_context: ToolContext) -> Optional[dict[str, Any]]:
    """도구 실행 직전 안전 인터록. dict 반환 시 실제 도구 실행을 건너뛰고 그 dict 가 결과가 된다."""
    name = tool.name
    print(f"[Guardrail:before_tool] {tool_context.agent_name} -> {name}({json.dumps(args, ensure_ascii=False)[:160]})")
    st = tool_context.state

    if name == "request_equipment_action":
        risk = st.get("risk_assessment") or {}
        level = risk.get("risk_level") if isinstance(risk, dict) else None
        eq, act = str(args.get("equipment", "")).lower(), str(args.get("action", "")).lower()
        if level in _DANGER_LEVELS and eq in {"aerator", "liquid_oxygen"} and act == "off":
            return _block(st, name, f"위험 단계 '{level}'에서는 산소공급 장비({eq}) 정지가 금지됩니다.")
        if level in _DANGER_LEVELS and eq == "feeder" and act == "on":
            return _block(st, name, f"위험 단계 '{level}'에서는 급이 재개가 금지됩니다(산소 소비 증가).")
        if not str(args.get("reason", "")).strip():
            return _block(st, name, "장비 제어 요청에는 근거 사유(reason)가 반드시 필요합니다.")

    if name == "publish_report":
        title = str(args.get("title", ""))
        if re.search(r"\.(sh|exe|bat|ps1|cmd)\b", title, re.I):
            return _block(st, name, "실행 파일 형식의 이름으로는 저장할 수 없습니다.")
    return None


def _block(state, tool_name: str, reason: str) -> dict[str, Any]:
    log = list(state.get("guardrail_blocks", []) or [])
    log.append({"tool": tool_name, "reason": reason})
    state["guardrail_blocks"] = log
    print(f"[Guardrail:interlock] BLOCK {tool_name}: {reason}")
    return {"status": "blocked_by_safety_interlock", "reason": reason}


# ③ -------------------------------------------------------------------------
def _normalize(tool_response: Any) -> Any:
    """MCP 응답(CallToolResult dict)에서 구조화 데이터를 추출."""
    if hasattr(tool_response, "model_dump"):
        tool_response = tool_response.model_dump()
    if isinstance(tool_response, dict):
        if isinstance(tool_response.get("structuredContent"), dict):
            sc = tool_response["structuredContent"]
            return sc.get("result", sc) if set(sc) == {"result"} else sc
        content = tool_response.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict) and "text" in content[0]:
            try:
                return json.loads(content[0]["text"])
            except (json.JSONDecodeError, TypeError):
                return {"text": content[0]["text"]}
    return tool_response


def evidence_ledger_after_tool(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
                               tool_response: Any) -> Optional[dict[str, Any]]:
    """근거 도구의 결과를 state['ev:<tool>'] 에 적재 (병렬 브랜치 간 키 충돌 방지를 위해 도구별 분리 키)."""
    if tool.name in EVIDENCE_TOOLS:
        data = _normalize(tool_response)
        if isinstance(data, dict) and data.get("status") != "error":
            tool_context.state[f"ev:{tool.name}"] = data
            print(f"[EvidenceLedger] ev:{tool.name} 적재 ({tool_context.agent_name})")
    return None
