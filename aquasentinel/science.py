"""결정론적 해양 과학 계산 모듈 (LLM이 아닌 코드가 수치를 계산한다).

- 용존산소 포화농도: Benson & Krause (1984) 식 — APHA Standard Methods 4500-O 에 수록된 형태
- 양식 위기지수(ARI, Aquaculture Risk Index): 본 프로젝트 자체 설계 지수 (공식 지표 아님)
"""

from __future__ import annotations

import math
from typing import Any


def do_saturation_mg_l(temp_c: float, salinity_psu: float) -> float:
    """수온·염분에서의 용존산소 포화농도(mg/L, 1기압)를 계산한다."""
    t = temp_c + 273.15
    ln_do = (
        -139.34411
        + 1.575701e5 / t
        - 6.642308e7 / t**2
        + 1.243800e10 / t**3
        - 8.621949e11 / t**4
    )
    ln_do -= salinity_psu * (1.7674e-2 - 1.0754e1 / t + 2.1407e3 / t**2)
    return math.exp(ln_do)


# ---------------------------------------------------------------------------
# ARI(양식 위기지수) — 시스템 설계값. 가중치는 교육용 가정이며 현장 검증 전 단계.
# ---------------------------------------------------------------------------
ARI_LEVELS = [(80, "심각"), (60, "경계"), (40, "주의"), (20, "관심"), (0, "정상")]


def _level(score: float) -> str:
    for cut, name in ARI_LEVELS:
        if score >= cut:
            return name
    return "정상"


def compute_ari(hypoxia: dict[str, Any] | None, heat: dict[str, Any] | None,
                weather: dict[str, Any] | None) -> dict[str, Any]:
    """세 가지 근거(빈산소·고수온·기상)로 0~100점 위기지수와 단계를 산출한다."""
    missing = [n for n, v in (("analyze_hypoxia", hypoxia), ("assess_heat_alert", heat),
                              ("get_weather_forecast", weather)) if not v]
    if missing:
        return {"status": "insufficient_evidence", "missing_evidence": missing,
                "ari_score": None, "risk_level": "판정불가"}

    # 1) 고수온 (최대 40점)
    heat_pts = {"경보": 40, "주의보": 30, "예비특보": 15}.get(heat["heat_alert_level"], 0)
    # 2) 빈산소 (최대 45점 = 최저DO 35 + 누적시간 10)
    do_min = hypoxia["do_min_mg_l"]
    if do_min <= 2.0:
        do_pts = 35
    elif do_min <= 3.0:
        do_pts = 28
    elif do_min <= 4.0:
        do_pts = 15
    elif do_min <= 5.0:
        do_pts = 5
    else:
        do_pts = 0
    dur_pts = min(10, hypoxia["hours_do_le_3"] // 3)
    # 3) 기상 (최대 15점) — 무풍일(성층 강화) + 폭염
    wx_pts = min(10, 4 * weather["calm_days_wind_lt_3ms"]) + (5 if weather["max_air_temp_c"] >= 33 else 0)

    score = min(100, heat_pts + do_pts + dur_pts + wx_pts)
    return {
        "status": "success",
        "ari_score": score,
        "risk_level": _level(score),
        "components": {
            "heat_points": heat_pts,
            "do_min_points": do_pts,
            "hypoxia_duration_points": dur_pts,
            "weather_points": wx_pts,
        },
        "level_cutoffs": "정상<20≤관심<40≤주의<60≤경계<80≤심각",
        "basis": {
            "heat_alert_level": heat["heat_alert_level"],
            "consecutive_days_ge_28c": heat["consecutive_days_ge_28c"],
            "latest_daily_max_c": heat["latest_daily_max_c"],
            "species": heat["species"],
            "species_upper_temp_c": heat["species_upper_temp_c"],
            "degree_hours_over_upper": heat["degree_hours_over_upper"],
            "do_min_mg_l": do_min,
            "do_min_time": hypoxia["do_min_time"],
            "latest_do_mg_l": hypoxia["latest_do_mg_l"],
            "hours_do_le_3": hypoxia["hours_do_le_3"],
            "longest_hypoxia_hours": hypoxia["longest_hypoxia_hours"],
            "calm_days_wind_lt_3ms": weather["calm_days_wind_lt_3ms"],
            "max_air_temp_c": weather["max_air_temp_c"],
        },
        "disclaimer": "ARI는 본 프로젝트 자체 설계 지수로, 기관 공식 위기경보 단계가 아님",
    }
