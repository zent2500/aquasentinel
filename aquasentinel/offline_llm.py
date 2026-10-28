"""OfflineScriptedLlm — Gemini API 없이 ADK 런타임 전체를 구동하기 위한 결정론적 BaseLlm.

목적
  - CI/회귀 테스트: 네트워크·API 키·과금 없이 에이전트 트리, 상태 전달, 도구 호출, 콜백, 루프 종료를
    매번 동일하게 재현한다. (LLM 응답 품질 평가는 Gemini 모드에서 별도로 수행)
  - 실행 방법: 환경 변수 AQUA_OFFLINE=1  →  agent.py 가 모든 LlmAgent 의 model 을 이 클래스로 교체

동작 원리
  - 에이전트별 '역할(role)' 스크립트가 llm_request 의 system_instruction(상태 주입 결과)과
    자신의 function_response 이력을 읽어 다음 function_call 또는 최종 텍스트를 생성한다.
  - 텍스트의 모든 수치는 도구 응답/상태에서만 가져온다(스크립트 자체에 수치를 하드코딩하지 않음).
  - 루프 동작 시연을 위해 action_planner 의 1회차 초안은 '모니터링 지표' 섹션을 의도적으로 누락하도록
    스크립트되어 있다 → grounding_critic 이 결정론 검증으로 반려 → 2회차에 보완 → exit_loop.
"""

from __future__ import annotations

import ast
import re
from typing import Any, AsyncGenerator

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

FARM_RE = re.compile(r"\b([A-Za-z]{2}-\d{2})\b")
CRISIS_KW = re.compile(r"진단|점검|위기|분석|대응|위험|상황|보고서")


def _instruction(req: LlmRequest) -> str:
    si = req.config.system_instruction if req.config else None
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    if isinstance(si, types.Content):
        return " ".join(p.text or "" for p in si.parts or [])
    return str(si)


def _tail_calls(req: LlmRequest) -> list[tuple[str, dict]]:
    """현재 턴에서 이 에이전트가 받은 function_response 목록 (끝에서부터 연속 구간)."""
    out: list[tuple[str, dict]] = []
    for c in reversed(req.contents or []):
        parts = c.parts or []
        frs = [p.function_response for p in parts if p.function_response]
        fcs = [p.function_call for p in parts if p.function_call]
        if frs:
            out.extend((fr.name, dict(fr.response or {})) for fr in frs)
        elif fcs:
            continue
        else:
            break
    out.reverse()
    return out


def _last_user_text(req: LlmRequest) -> str:
    for c in reversed(req.contents or []):
        if c.role == "user":
            t = " ".join(p.text for p in c.parts or [] if p.text)
            if t and not t.startswith("For context"):
                return t
    return ""


def _resp(r: dict) -> dict:
    """MCP 응답이면 structuredContent 를 꺼낸다."""
    if isinstance(r.get("structuredContent"), dict):
        return r["structuredContent"]
    if "result" in r and isinstance(r["result"], dict):
        return r["result"]
    content = r.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        import json
        try:
            return json.loads(content[0].get("text", "{}"))
        except Exception:
            pass
    return r


def _state_value(instr: str, label: str) -> Any:
    m = re.search(re.escape(label) + r"\s*(\{.*?\}|\[.*?\])\s*$", instr, flags=re.M | re.S)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return None


def _call(name: str, **args: Any) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name=name, args=args))]))


def _text(t: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=t)]))


class OfflineScriptedLlm(BaseLlm):
    role: str = "generic"

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"offline/.*"]

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        handler = getattr(self, f"_role_{self.role}", None)
        yield handler(llm_request) if handler else _text("(offline) 지원하지 않는 역할입니다.")

    # ------------------------------------------------------------------ roles
    def _role_coordinator(self, req: LlmRequest) -> LlmResponse:
        tail = _tail_calls(req)
        if tail:
            name, r = tail[-1]
            if name == "register_incident":
                if r.get("status") == "registered":
                    return _call("transfer_to_agent", agent_name="crisis_pipeline")
                ids = ", ".join(r.get("available_farm_ids", []))
                return _text(f"요청하신 양식장을 찾을 수 없습니다. ({r.get('error')})\n"
                             f"등록된 양식장 ID: {ids}\n예) '양식장 GH-01 위기 진단해줘'")
            return _text("요청을 처리했습니다.")
        text = _last_user_text(req)
        m = FARM_RE.search(text)
        if m and CRISIS_KW.search(text):
            return _call("register_incident", farm_id=m.group(1).upper(), user_request=text)
        if re.search(r"목록|기준|특보|요령|어떤 양식장|알려|무엇|뭐", text):
            return _call("transfer_to_agent", agent_name="farm_qa_agent")
        return _text("안녕하세요, 해상 양식장 위기대응 관제센터 **AquaSentinel** 입니다.\n"
                     "- 예시 1) `양식장 GH-01 위기 진단해줘`\n- 예시 2) `고수온 특보 기준 알려줘`")

    def _role_farm_qa(self, req: LlmRequest) -> LlmResponse:
        tail = _tail_calls(req)
        text = _last_user_text(req)
        if not tail:
            if FARM_RE.search(text) and CRISIS_KW.search(text):
                return _call("transfer_to_agent", agent_name="aqua_coordinator")
            if re.search(r"목록|어떤 양식장", text):
                return _call("list_farms")
            m = FARM_RE.search(text)
            if m and "특보" in text:
                return _call("get_official_alerts", farm_id=m.group(1).upper())
            if m:
                return _call("get_farm_profile", farm_id=m.group(1).upper())
            return _call("lookup_response_manual", query=text, top_k=3)
        name, r = tail[-1]
        r = _resp(r)
        if name == "list_farms":
            rows = "\n".join(f"| {f['farm_id']} | {f['name']} | {f['region']} | {f['farm_type']} | {f['species']} |"
                             for f in r.get("farms", []))
            return _text("모니터링 중인 양식장 목록입니다.\n\n| ID | 이름 | 지역 | 형태 | 품종 |\n|---|---|---|---|---|\n"
                         + rows + "\n\n위기 진단: `양식장 GH-01 위기 진단해줘`")
        if name == "lookup_response_manual":
            lines = [f"- {h['text']} [{h['id']}]\n  - 출처: {h['source']}" for h in r.get("results", [])]
            return _text("지식베이스 검색 결과입니다.\n" + "\n".join(lines) if lines else "관련 근거를 찾지 못했습니다.")
        if name == "get_official_alerts":
            al = r.get("alerts", [])
            body = "\n".join(f"- {a['type']} {a['level']} ({a['issued_at']}, {a['issuer']})" for a in al) or "- 현재 특보 없음"
            return _text(f"{r.get('sea_area')} 해역 특보\n{body}\n\n※ {r.get('disclaimer', '')}")
        if r.get("status") == "error":
            return _text(f"{r.get('error')} — 등록 ID: {', '.join(r.get('available_farm_ids', []))}")
        p = r.get("profile", {})
        return _text(f"**{p.get('name')}** ({p.get('farm_id')})\n- 지역/해역: {p.get('region')} / {p.get('sea_area')}\n"
                     f"- 형태·품종: {p.get('farm_type')} · {p.get('species')}\n- 보유 장비: {', '.join(p.get('equipment', []))}")

    def _farm(self, req: LlmRequest) -> str:
        m = re.search(r"대상 양식장:\s*([A-Z]{2}-\d{2})", _instruction(req))
        return m.group(1) if m else "GH-01"

    def _role_water(self, req: LlmRequest) -> LlmResponse:
        tail = dict(_tail_calls(req))
        fid = self._farm(req)
        if "analyze_hypoxia" not in tail:
            return _call("analyze_hypoxia", farm_id=fid, hours=72)
        if "get_sensor_timeseries" not in tail:
            return _call("get_sensor_timeseries", farm_id=fid, hours=72)
        h = _resp(tail["analyze_hypoxia"])
        if "compute_do_saturation" not in tail:
            return _call("compute_do_saturation", temperature_c=h["latest_water_temp_c"],
                         salinity_psu=h["latest_salinity_psu"], measured_do_mg_l=h["latest_do_mg_l"])
        s = _resp(tail["compute_do_saturation"])
        judge = "빈산소 발생 — 새벽 산소공급 필요" if h["hours_do_le_3"] > 0 else "빈산소 미발생"
        return _text(f"[수질 분석 · {fid}]\n- 최저 DO {h['do_min_mg_l']} mg/L ({h['do_min_time']})\n"
                     f"- DO≤3 mg/L 누적 {h['hours_do_le_3']}시간, 최장 연속 {h['longest_hypoxia_hours']}시간\n"
                     f"- 최신 DO {s['measured_do_mg_l']} mg/L = 포화농도 {s['do_saturation_mg_l']} mg/L 대비 {s['do_saturation_percent']}%\n"
                     f"- 판단: {judge}")

    def _role_thermal(self, req: LlmRequest) -> LlmResponse:
        tail = dict(_tail_calls(req))
        fid = self._farm(req)
        if "assess_heat_alert" not in tail:
            return _call("assess_heat_alert", farm_id=fid)
        h = _resp(tail["assess_heat_alert"])
        trend = " → ".join(f"{v}℃" for v in h["daily_max_temp_c"].values())
        return _text(f"[고수온 분석 · {fid}]\n- 일 최고수온: {trend}\n- 28℃ 이상 연속 {h['consecutive_days_ge_28c']}일 → 판정: **{h['heat_alert_level']}**\n"
                     f"- {h['species']} 관리상한 {h['species_upper_temp_c']}℃ 초과 누적 {h['degree_hours_over_upper']} 도·시")

    def _role_weather(self, req: LlmRequest) -> LlmResponse:
        tail = dict(_tail_calls(req))
        fid = self._farm(req)
        if "get_weather_forecast" not in tail:
            return _call("get_weather_forecast", farm_id=fid)
        if "get_official_alerts" not in tail:
            return _call("get_official_alerts", farm_id=fid)
        w, a = _resp(tail["get_weather_forecast"]), _resp(tail["get_official_alerts"])
        al = ", ".join(f"{x['type']} {x['level']}" for x in a.get("alerts", [])) or "없음"
        return _text(f"[기상·특보 분석 · {fid}]\n- 3일 최고기온 {w['max_air_temp_c']}℃, 무풍일(3 m/s 미만) {w['calm_days_wind_lt_3ms']}일\n"
                     f"- 강수: " + ", ".join(f"{d['date'][5:]} {d['rain_mm']}mm" for d in w["forecast"]) +
                     f"\n- 해역 특보({a.get('sea_area')}): {al}")

    def _role_planner(self, req: LlmRequest) -> LlmResponse:
        instr = _instruction(req)
        tail = _tail_calls(req)
        done = {n for n, _ in tail}
        risk = _state_value(instr, "RiskGate 결정론 판정(JSON):") or {}
        level = risk.get("risk_level", "판정불가")
        b = risk.get("basis", {})
        first_round = "초기 작성 단계" in instr
        danger = level in {"주의", "경계", "심각"}
        if "lookup_response_manual" not in done:
            q = ("빈산소 산소공급 급이 중단 고수온 관리요령 새벽 감시" if danger
                 else "정상 단계 예방 관리 센서 점검 새벽 감시")
            return _call("lookup_response_manual", query=q, top_k=5)
        kb = _resp(dict(tail)["lookup_response_manual"]).get("results", [])
        ids = [h["id"] for h in kb]
        if first_round and danger and b.get("hours_do_le_3", 0) > 0:
            if ("request_equipment_action", "aerator") not in {(n, r.get("command", {}).get("equipment")) for n, r in tail}:
                return _call("request_equipment_action", equipment="aerator", action="on",
                             reason=f"최저 DO {b.get('do_min_mg_l')} mg/L, DO≤3 mg/L 누적 {b.get('hours_do_le_3')}시간")
            if ("request_equipment_action", "feeder") not in {(n, r.get("command", {}).get("equipment")) for n, r in tail}:
                return _call("request_equipment_action", equipment="feeder", action="off",
                             reason=f"고수온 {b.get('heat_alert_level')} + 빈산소 중첩, 급이 중단")
        cite = lambda *c: " ".join(f"[{x}]" for x in c if x in ids) or (f"[{ids[0]}]" if ids else "")
        lines = [
            "## 1. 상황 요약",
            f"- {b.get('species')} 양식장의 일 최고수온 {b.get('latest_daily_max_c')}℃, 28℃ 이상 {b.get('consecutive_days_ge_28c')}일 지속(고수온 {b.get('heat_alert_level')}).",
            f"- 최저 DO {b.get('do_min_mg_l')} mg/L({b.get('do_min_time')}), DO≤3 mg/L 누적 {b.get('hours_do_le_3')}시간·최장 연속 {b.get('longest_hypoxia_hours')}시간.",
            f"- 향후 3일 최고기온 {b.get('max_air_temp_c')}℃, 무풍일 {b.get('calm_days_wind_lt_3ms')}일.",
            "", "## 2. 위험도 판정",
            f"- RiskGate 결정론 판정: **{level}** (ARI {risk.get('ari_score')}점, 구간 {risk.get('level_cutoffs')})",
            "", "## 3. 즉시 조치",
        ]
        if danger:
            lines += [f"1. 산소발생기·액화산소 공급장치 가동, 가동 중 장비 임의 정지 금지 {cite('KB-05')}",
                      f"2. 급이 중단(회복 후 소량부터 재개) {cite('KB-06', 'KB-03')}",
                      f"3. 사육 밀도 조정·조기 출하 검토 {cite('KB-03')}"]
        else:
            lines += [f"1. 정기 센서 점검과 대응 장비 시운전 {cite('KB-09')}",
                      f"2. 사육 밀도 점검으로 예비특보 이전 대비 {cite('KB-09')}"]
        if not first_round:
            lines += ["", "## 4. 모니터링 지표",
                      f"- 새벽 시간대 DO가 빈산소 기준 3 mg/L 이하인지 1시간 간격 재측정 {cite('KB-05', 'KB-04')}",
                      f"- 오후 최고수온 28℃ 지속 여부와 무풍일 여부 {cite('KB-07', 'KB-02')}"]
        lines += ["", "## 5. 근거",
                  "- MCP 도구: analyze_hypoxia, assess_heat_alert, get_weather_forecast, get_official_alerts",
                  "- 계산 도구: compute_do_saturation(Benson-Krause), RiskGateAgent(ARI)",
                  "- 지식베이스: " + ", ".join(f"{h['id']}({h['source']})" for h in kb)]
        return _text("\n".join(lines))

    def _role_critic(self, req: LlmRequest) -> LlmResponse:
        tail = dict(_tail_calls(req))
        if "verify_numeric_grounding" not in tail:
            return _call("verify_numeric_grounding")
        r = tail["verify_numeric_grounding"]
        if r.get("passed"):
            if "exit_loop" not in tail:
                return _call("exit_loop")
            return _text(f"✅ 검증 통과 — 수치 {r['checked_numeric_claims']}건 모두 근거 원장과 일치, 루프 종료")
        fb = ["❌ 검증 미통과 — 아래 항목을 수정하세요."]
        if r.get("missing_sections"):
            fb.append(f"- 누락 섹션: {r['missing_sections']}")
        if r.get("ungrounded_claims"):
            fb.append(f"- 근거 없는 수치: {r['ungrounded_claims']}")
        if r.get("invalid_citations"):
            fb.append(f"- 존재하지 않는 인용: {r['invalid_citations']}")
        if not r.get("risk_level_consistent"):
            fb.append(f"- 위험 단계 표기를 '{r.get('risk_level_expected')}'로 일치시킬 것")
        return _text("\n".join(fb))

    def _role_publisher(self, req: LlmRequest) -> LlmResponse:
        instr = _instruction(req)
        tail = dict(_tail_calls(req))
        if "render_trend_chart" not in tail:
            return _call("render_trend_chart")
        m = re.search(r"\[검증 통과 초안\]\n(.*?)\n\[검증 결과\]", instr, flags=re.S)
        draft = m.group(1).strip() if m else ""
        if "publish_report" not in tail:
            cmds = _state_value(instr, "[장비 제어 요청 기록]") or []
            table = "\n\n## 6. 장비 제어 요청 기록\n| 장비 | 동작 | 사유 | 상태 |\n|---|---|---|---|\n" + (
                "\n".join(f"| {c['equipment']} | {c['action']} | {c['reason']} | {c['status']} |" for c in cmds)
                or "| - | - | 요청 없음 | - |")
            return _call("publish_report", title="AquaSentinel 위기대응 보고서", markdown_body=draft + table)
        p, ch = tail["publish_report"], tail["render_trend_chart"]
        if p.get("status") not in ("published", "saved_file_only"):
            return _text(f"보고서를 발행하지 못했습니다: {p.get('reason')}")
        lvl = re.search(r"\*\*(\S+)\*\* \(ARI (\d+)점", draft)
        acts = re.findall(r"^\d\. (.+?) \[", draft, flags=re.M)[:3]
        return _text(f"📄 **최종 보고서 발행 완료**\n- 위험 단계: **{lvl.group(1) if lvl else '-'}** (ARI {lvl.group(2) if lvl else '-'}점)\n"
                     + "\n".join(f"- 조치: {a}" for a in acts) +
                     f"\n- 보고서 Artifact: `{p.get('artifact')}`\n- 추세 차트 Artifact: `{ch.get('artifact')}`")
