"""AquaSentinel 커스텀 FunctionTool 모음 (ToolContext 기반 상태 읽기/쓰기).

도구 목록
  1. register_incident          : 사건 접수 + 세션 상태 초기화 (Coordinator)
  2. compute_do_saturation      : 용존산소 포화농도/포화도 결정론 계산 (수질 분석가)
  3. lookup_response_manual     : 대응 매뉴얼 BM25 검색 RAG (대응계획 수립가, Q&A)
  4. request_equipment_action   : 산소발생기·급이기 등 장비 제어 요청(가상 큐) — 안전 인터록 대상
  5. verify_numeric_grounding   : 초안 속 모든 수치·인용의 근거 대조 (Grounding Critic)
  6. exit_loop                  : LoopAgent 종료
  7. render_trend_chart         : 수온·DO 추세 차트 PNG → ADK Artifact 저장
  8. publish_report             : 최종 보고서 .md 파일 + ADK Artifact 저장
"""

from __future__ import annotations

import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from google.adk.tools.tool_context import ToolContext
from google.genai import types

from . import knowledge_base, science

# 양식장 대장(CMDB)의 ID 목록 — MCP 서버의 FARMS 와 동일하게 유지
KNOWN_FARM_IDS = {"GH-01", "WD-02", "TY-03"}
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
REQUIRED_SECTIONS = ["상황 요약", "위험도", "즉시 조치", "모니터링", "근거"]
_UNIT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(℃|°C|mg/L|mg/l|%|시간|h\b|일|m/s|mm|점|도·시)")


# ---------------------------------------------------------------------------
# 1. 사건 접수
# ---------------------------------------------------------------------------
def register_incident(farm_id: str, user_request: str, tool_context: ToolContext) -> dict[str, Any]:
    """위기 진단 요청을 접수하고 파이프라인이 사용할 세션 상태를 초기화합니다.

    Args:
        farm_id: 진단할 양식장 ID (예: 'GH-01', 'WD-02', 'TY-03').
        user_request: 사용자의 원 요청 문장.
    """
    fid = farm_id.strip().upper()
    if fid not in KNOWN_FARM_IDS:
        return {"status": "error", "error": f"등록되지 않은 양식장 ID: {farm_id}",
                "available_farm_ids": sorted(KNOWN_FARM_IDS)}
    incident_id = f"INC-{fid}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    st = tool_context.state
    st["incident_id"] = incident_id
    st["incident_farm_id"] = fid
    st["incident_request"] = user_request
    st["critic_feedback"] = "초기 작성 단계 (피드백 없음)"
    st["action_plan_draft"] = ""
    st["loop_iteration"] = 0
    st["kb_retrieved"] = []
    st["equipment_commands"] = []
    return {"status": "registered", "incident_id": incident_id, "farm_id": fid}


# ---------------------------------------------------------------------------
# 2. 용존산소 포화농도 (결정론 계산)
# ---------------------------------------------------------------------------
def compute_do_saturation(temperature_c: float, salinity_psu: float, measured_do_mg_l: float,
                          tool_context: ToolContext) -> dict[str, Any]:
    """수온·염분으로 용존산소 포화농도(Benson-Krause 식)와 실측 DO의 포화도(%)를 계산합니다.

    Args:
        temperature_c: 수온(℃).
        salinity_psu: 염분(psu).
        measured_do_mg_l: 실측 용존산소(mg/L).
    """
    sat = science.do_saturation_mg_l(temperature_c, salinity_psu)
    pct = measured_do_mg_l / sat * 100
    return {
        "status": "success",
        "do_saturation_mg_l": round(sat, 2),
        "measured_do_mg_l": measured_do_mg_l,
        "do_saturation_percent": round(pct, 1),
        "method": "Benson & Krause (1984), APHA 4500-O",
        "inputs": {"temperature_c": temperature_c, "salinity_psu": salinity_psu},
    }


# ---------------------------------------------------------------------------
# 3. 매뉴얼 RAG
# ---------------------------------------------------------------------------
def lookup_response_manual(query: str, tool_context: ToolContext, top_k: int = 4) -> dict[str, Any]:
    """양식장 위기대응 지식베이스를 BM25로 검색해 근거 청크(KB-xx)를 반환합니다.

    Args:
        query: 검색 질의 (예: '빈산소 산소공급 급이 중단').
        top_k: 반환할 청크 수 (1~6).
    """
    hits = knowledge_base.search(query, top_k=max(1, min(int(top_k), 6)))
    retrieved = list(tool_context.state.get("kb_retrieved", []) or [])
    for h in hits:
        if h["id"] not in retrieved:
            retrieved.append(h["id"])
    tool_context.state["kb_retrieved"] = retrieved
    return {"status": "success", "query": query, "results": hits}


# ---------------------------------------------------------------------------
# 4. 장비 제어 요청 (가상 제어 큐) — callbacks.safety_interlock 이 사전 차단
# ---------------------------------------------------------------------------
def request_equipment_action(equipment: str, action: str, reason: str,
                             tool_context: ToolContext) -> dict[str, Any]:
    """양식장 장비 제어를 요청합니다(가상 제어 큐에 등록, 실제 장비 미연동).

    Args:
        equipment: 장비 종류 ('aerator' 산소발생기, 'liquid_oxygen' 액화산소, 'feeder' 급이기, 'shade_net' 차광막).
        action: 'on' 또는 'off'.
        reason: 요청 사유(근거 수치 포함).
    """
    cmds = list(tool_context.state.get("equipment_commands", []) or [])
    cmd = {"equipment": equipment, "action": action, "reason": reason,
           "farm_id": tool_context.state.get("incident_farm_id"),
           "queued_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": "queued(simulated)"}
    cmds.append(cmd)
    tool_context.state["equipment_commands"] = cmds
    return {"status": "queued", "command": cmd}


# ---------------------------------------------------------------------------
# 5. 수치 그라운딩 검증 (결정론적 Critic 도구)
# ---------------------------------------------------------------------------
def _collect_numbers(obj: Any, out: list[float], skip_keys: set[str]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip_keys:
                continue
            _collect_numbers(v, out, skip_keys)
            if isinstance(k, str):
                _collect_numbers(k, out, skip_keys)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out, skip_keys)


def _as_dict(state: Any) -> dict[str, Any]:
    return state.to_dict() if hasattr(state, "to_dict") else dict(state)


def evidence_numbers(state: Any) -> list[float]:
    state = _as_dict(state)
    nums: list[float] = []
    skip = {"series", "timestamp", "issued_at", "queued_at", "incident_id"}
    for k in list(state.keys()):
        if k.startswith("ev:") or k == "risk_assessment":
            _collect_numbers(state[k], nums, skip)
    for cid in state.get("kb_retrieved", []) or []:
        for c in knowledge_base.KB_CHUNKS:
            if c["id"] == cid:
                _collect_numbers(c["text"], nums, skip)
    return nums


def verify_numeric_grounding(tool_context: ToolContext) -> dict[str, Any]:
    """대응계획 초안(state['action_plan_draft'])의 수치·인용·필수 섹션을 근거 원장과 대조 검증합니다.

    검증 항목: (1) 단위가 붙은 모든 수치가 도구 산출 근거값과 일치하는가 (2) [KB-xx] 인용이 실제 검색된
    청크인가 (3) 필수 섹션 5개 포함 여부 (4) 위험 단계 표기가 RiskGate 판정과 일치하는가.
    """
    st = tool_context.state
    draft = str(st.get("action_plan_draft", "") or "")
    nums = evidence_numbers(st)
    claims = _UNIT_RE.findall(draft)
    ungrounded = []
    for val, unit in claims:
        x = float(val)
        tol = max(0.05, abs(x) * 0.01)
        if not any(abs(x - e) <= tol for e in nums):
            ungrounded.append(f"{val}{unit}")
    cited = set(re.findall(r"KB-\d{2}", draft))
    retrieved = set(st.get("kb_retrieved", []) or [])
    invalid = sorted(c for c in cited if c not in retrieved)
    missing = [s for s in REQUIRED_SECTIONS if s not in draft]
    risk = st.get("risk_assessment") or {}
    level = risk.get("risk_level") if isinstance(risk, dict) else None
    level_ok = bool(level) and level in draft
    passed = bool(draft) and not ungrounded and not invalid and not missing and level_ok and bool(cited)
    report = {
        "passed": passed,
        "checked_numeric_claims": len(claims),
        "ungrounded_claims": sorted(set(ungrounded)),
        "invalid_citations": invalid,
        "has_citation": bool(cited),
        "missing_sections": missing,
        "risk_level_expected": level,
        "risk_level_consistent": level_ok,
    }
    st["grounding_report"] = report
    st["loop_iteration"] = int(st.get("loop_iteration", 0) or 0) + 1
    return report


# ---------------------------------------------------------------------------
# 6. 루프 종료
# ---------------------------------------------------------------------------
def exit_loop(tool_context: ToolContext) -> dict[str, Any]:
    """그라운딩 검증을 통과했을 때만 호출하여 작성-검증 LoopAgent를 종료합니다."""
    tool_context.actions.escalate = True
    return {"status": "loop_terminated", "agent": tool_context.agent_name}


# ---------------------------------------------------------------------------
# 7. 추세 차트 → Artifact
# ---------------------------------------------------------------------------
async def render_trend_chart(tool_context: ToolContext) -> dict[str, Any]:
    """센서 시계열(수온·용존산소)을 차트 PNG로 그려 ADK Artifact로 저장합니다."""
    ev = tool_context.state.get("ev:get_sensor_timeseries")
    if not ev or not ev.get("series"):
        return {"status": "error", "error": "센서 시계열 근거(ev:get_sensor_timeseries)가 없습니다."}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fam in ("Malgun Gothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "AppleGothic"):
        if any(fam == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams["axes.unicode_minus"] = False

    s = ev["series"]
    x = list(range(len(s)))
    fig, ax1 = plt.subplots(figsize=(9, 3.6), dpi=120)
    ax1.plot(x, [r["water_temp_c"] for r in s], color="#d1495b", lw=1.8, label="수온(℃)")
    ax1.axhline(28, color="#d1495b", ls=":", lw=1)
    ax1.set_ylabel("수온 (℃)")
    ax2 = ax1.twinx()
    ax2.plot(x, [r["do_mg_l"] for r in s], color="#00798c", lw=1.8, label="DO(mg/L)")
    ax2.axhline(3, color="#00798c", ls=":", lw=1)
    ax2.set_ylabel("용존산소 (mg/L)")
    ticks = x[::12]
    ax1.set_xticks(ticks, [s[i]["timestamp"][5:13] for i in ticks], rotation=30, fontsize=7)
    ax1.set_title(f"{ev.get('farm_id')} 최근 {len(s)}시간 수온·용존산소 (점선: 28℃ / 3 mg/L)", fontsize=10)
    for spine in ("top",):
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    png = buf.getvalue()
    name = f"{ev.get('farm_id')}_trend.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / name).write_bytes(png)
    version = None
    try:
        version = await tool_context.save_artifact(name, types.Part.from_bytes(data=png, mime_type="image/png"))
    except Exception as e:  # artifact service 미구성 환경
        return {"status": "saved_file_only", "file": str(OUTPUT_DIR / name), "artifact_error": str(e)}
    return {"status": "success", "artifact": name, "version": version, "file": str(OUTPUT_DIR / name)}


# ---------------------------------------------------------------------------
# 8. 최종 보고서 → 파일 + Artifact
# ---------------------------------------------------------------------------
async def publish_report(title: str, markdown_body: str, tool_context: ToolContext) -> dict[str, Any]:
    """검증을 통과한 최종 대응 보고서를 Markdown 파일과 ADK Artifact로 발행합니다.

    Args:
        title: 보고서 제목.
        markdown_body: 보고서 본문(Markdown).
    """
    st = tool_context.state
    report = st.get("grounding_report") or {}
    if not report.get("passed"):
        return {"status": "rejected", "reason": "그라운딩 검증을 통과하지 못한 초안은 발행할 수 없습니다.",
                "grounding_report": report}
    header = (f"# {title}\n\n- 사건 ID: {st.get('incident_id')}\n- 양식장: {st.get('incident_farm_id')}\n"
              f"- 발행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
              f"- 그라운딩 검증: 통과 (수치 {report.get('checked_numeric_claims')}건 대조)\n\n")
    ledger = {k: v for k, v in _as_dict(st).items() if k.startswith("ev:") and k != "ev:get_sensor_timeseries"}
    appendix = ("\n\n---\n## 부록. 근거 원장(Evidence Ledger)\n```json\n"
                + json.dumps({"risk_assessment": st.get("risk_assessment"), **ledger},
                             ensure_ascii=False, indent=2) + "\n```\n")
    text = header + markdown_body + appendix
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{st.get('incident_id', 'report')}.md"
    (OUTPUT_DIR / fname).write_text(text, encoding="utf-8")
    st["published_report_path"] = str(OUTPUT_DIR / fname)
    version = None
    try:
        version = await tool_context.save_artifact(fname, types.Part.from_bytes(
            data=text.encode("utf-8"), mime_type="text/markdown"))
    except Exception as e:
        return {"status": "saved_file_only", "file": str(OUTPUT_DIR / fname), "artifact_error": str(e)}
    return {"status": "published", "file": str(OUTPUT_DIR / fname), "artifact": fname, "version": version}
