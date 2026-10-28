"""AquaSentinel FastMCP 데이터 서버.

양식장 대장·센서 시계열·기상예보·기관 특보를 MCP 도구로 제공하고,
수치 판정(빈산소 구간, 고수온 특보 단계)은 LLM이 아니라 **이 서버의 결정론적 코드**가 계산한다.

실행 방법
  # (1) stdio 모드: ADK가 자식 프로세스로 자동 실행 (기본값, 별도 실행 불필요)
  # (2) HTTP 모드: Cloud Run / 로컬 서버로 띄우고 AQUA_MCP_URL 로 연결
  python mcp_server/server.py --http          # 기본 포트 8080, /mcp

판정 기준 출처 (보고서 참고문헌 참조)
  - 고수온 예비특보: 수온 25℃ 도달(예상)  / 주의보: 28℃ 도달(예상) / 경보: 28℃ 3일 이상 지속(예상)
  - 빈산소수괴: 용존산소 3 mg/L 이하
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastmcp import FastMCP  # noqa: E402

import farm_data as db  # noqa: E402

mcp = FastMCP("AquaSentinel-Ocean-Data")

HEAT_PRE_ALERT_C = 25.0   # 예비특보
HEAT_ALERT_C = 28.0       # 주의보 / 경보 기준 수온
HEAT_WARNING_DAYS = 3     # 경보: 28℃ 이상 3일 이상 지속
HYPOXIA_DO_MG_L = 3.0     # 빈산소수괴 기준


def _farm_or_error(farm_id: str) -> dict[str, Any] | None:
    return db.FARMS.get(farm_id.strip().upper())


def _not_found(farm_id: str) -> dict[str, Any]:
    return {
        "status": "error",
        "error": f"등록되지 않은 양식장 ID: {farm_id}",
        "available_farm_ids": sorted(db.FARMS),
    }


@mcp.tool()
def list_farms() -> dict[str, Any]:
    """모니터링 대상 양식장 목록을 조회합니다.

    Returns:
        farm_id, 이름, 지역, 양식 형태, 품종을 담은 목록.
    """
    return {
        "status": "success",
        "farms": [
            {k: f[k] for k in ("farm_id", "name", "region", "farm_type", "species")}
            for f in db.FARMS.values()
        ],
    }


@mcp.tool()
def get_farm_profile(farm_id: str) -> dict[str, Any]:
    """양식장 대장(품종, 사육밀도, 보유 장비, 품종별 관리 상한수온)을 조회합니다.

    Args:
        farm_id: 양식장 ID (예: 'GH-01').
    """
    farm = _farm_or_error(farm_id)
    if not farm:
        return _not_found(farm_id)
    return {"status": "success", "profile": farm}


@mcp.tool()
def get_sensor_timeseries(farm_id: str, hours: int = 48) -> dict[str, Any]:
    """양식장 부이 센서의 시간별 수온·용존산소(DO)·염분 시계열을 조회합니다.

    Args:
        farm_id: 양식장 ID (예: 'GH-01').
        hours: 최근 몇 시간 데이터를 가져올지 (1~96).
    """
    farm = _farm_or_error(farm_id)
    if not farm:
        return _not_found(farm_id)
    hours = max(1, min(int(hours), db.HOURS))
    rows = db.SENSOR_DB[farm["farm_id"]][-hours:]
    return {
        "status": "success",
        "farm_id": farm["farm_id"],
        "reference_time": db.REFERENCE_TIME.strftime("%Y-%m-%d %H:%M"),
        "hours": hours,
        "series": rows,
    }


@mcp.tool()
def analyze_hypoxia(farm_id: str, hours: int = 72) -> dict[str, Any]:
    """용존산소 시계열에서 빈산소(DO ≤ 3 mg/L) 발생 구간을 결정론적으로 계산합니다.

    Args:
        farm_id: 양식장 ID (예: 'GH-01').
        hours: 분석 구간(시간), 기본 72.

    Returns:
        최저/평균 DO, 3 mg/L·4 mg/L 이하 누적 시간, 최장 연속 빈산소 시간, 최저 DO 발생 시각,
        최신 수온/염분(산소포화도 계산 입력값).
    """
    farm = _farm_or_error(farm_id)
    if not farm:
        return _not_found(farm_id)
    rows = db.SENSOR_DB[farm["farm_id"]][-max(1, min(int(hours), db.HOURS)):]
    dos = [r["do_mg_l"] for r in rows]
    below3 = [d <= HYPOXIA_DO_MG_L for d in dos]
    longest = cur = 0
    for flag in below3:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    i_min = min(range(len(dos)), key=dos.__getitem__)
    latest = rows[-1]
    return {
        "status": "success",
        "farm_id": farm["farm_id"],
        "window_hours": len(rows),
        "do_min_mg_l": round(min(dos), 2),
        "do_mean_mg_l": round(sum(dos) / len(dos), 2),
        "do_min_time": rows[i_min]["timestamp"],
        "hours_do_le_3": sum(below3),
        "hours_do_le_4": sum(d <= 4.0 for d in dos),
        "longest_hypoxia_hours": longest,
        "hypoxia_threshold_mg_l": HYPOXIA_DO_MG_L,
        "latest_water_temp_c": latest["water_temp_c"],
        "latest_salinity_psu": latest["salinity_psu"],
        "latest_do_mg_l": latest["do_mg_l"],
    }


@mcp.tool()
def assess_heat_alert(farm_id: str) -> dict[str, Any]:
    """일 최고수온과 연속 지속일수로 고수온 특보 단계(예비특보/주의보/경보)를 판정합니다.

    Args:
        farm_id: 양식장 ID (예: 'GH-01').

    Returns:
        일자별 최고수온, 28℃ 이상 연속일수, 판정 단계, 품종 관리상한 초과 누적 도·시(degree-hours).
    """
    farm = _farm_or_error(farm_id)
    if not farm:
        return _not_found(farm_id)
    rows = db.SENSOR_DB[farm["farm_id"]]
    daily: dict[str, float] = {}
    for r in rows:
        d = r["timestamp"][:10]
        daily[d] = max(daily.get(d, -99.0), r["water_temp_c"])
    days = sorted(daily)
    streak = 0
    for d in reversed(days):
        if daily[d] >= HEAT_ALERT_C:
            streak += 1
        else:
            break
    latest_max = daily[days[-1]]
    if streak >= HEAT_WARNING_DAYS:
        level = "경보"
    elif latest_max >= HEAT_ALERT_C:
        level = "주의보"
    elif latest_max >= HEAT_PRE_ALERT_C:
        level = "예비특보"
    else:
        level = "해당없음"
    upper = farm["species_upper_temp_c"]
    degree_hours = sum(max(0.0, r["water_temp_c"] - upper) for r in rows)
    return {
        "status": "success",
        "farm_id": farm["farm_id"],
        "daily_max_temp_c": {d: round(daily[d], 2) for d in days},
        "latest_daily_max_c": round(latest_max, 2),
        "consecutive_days_ge_28c": streak,
        "heat_alert_level": level,
        "criteria": "예비특보 25℃ 도달 / 주의보 28℃ 도달 / 경보 28℃ 3일 이상 지속",
        "species": farm["species"],
        "species_upper_temp_c": upper,
        "degree_hours_over_upper": round(degree_hours, 1),
    }


@mcp.tool()
def get_weather_forecast(farm_id: str) -> dict[str, Any]:
    """양식장 해역의 향후 3일 기상예보(최고기온, 풍속, 강수량)를 조회합니다.

    Args:
        farm_id: 양식장 ID (예: 'GH-01').
    """
    farm = _farm_or_error(farm_id)
    if not farm:
        return _not_found(farm_id)
    fc = db.WEATHER_FORECAST[farm["farm_id"]]
    calm_days = sum(1 for d in fc if d["wind_speed_ms"] < 3.0)
    return {
        "status": "success",
        "farm_id": farm["farm_id"],
        "forecast": fc,
        "max_air_temp_c": max(d["air_temp_max_c"] for d in fc),
        "calm_days_wind_lt_3ms": calm_days,
        "note": "풍속 3 m/s 미만 무풍일은 수층 성층 강화로 저층 빈산소 위험을 키우는 요인으로 본다(시스템 설계 가정).",
    }


@mcp.tool()
def get_official_alerts(farm_id: str) -> dict[str, Any]:
    """양식장이 속한 해역의 기관 특보·속보(가상 데이터)를 조회합니다.

    Args:
        farm_id: 양식장 ID (예: 'GH-01').
    """
    farm = _farm_or_error(farm_id)
    if not farm:
        return _not_found(farm_id)
    return {
        "status": "success",
        "farm_id": farm["farm_id"],
        "sea_area": farm["sea_area"],
        "alerts": db.OFFICIAL_ALERTS.get(farm["sea_area"], []),
        "disclaimer": "교육용 가상 특보 데이터입니다.",
    }


if __name__ == "__main__":
    if "--http" in sys.argv:
        port = int(os.getenv("PORT", "8080"))
        mcp.run(transport="http", host="0.0.0.0", port=port)
    else:
        mcp.run()  # stdio
