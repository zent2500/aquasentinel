"""양식장 센서·특보·기상 Mock 데이터셋 (교육용 가상 데이터).

※ 모든 값은 실습용으로 합성(synthetic)한 가상 데이터이며 실제 관측값이 아닙니다.
   난수 시드를 고정해 매 실행마다 동일한 값이 생성되므로 테스트 재현성이 보장됩니다.

시나리오
- GH-01 : 고흥 해상가두리(조피볼락) — 28℃ 이상 고수온 4일 지속 + 야간 빈산소(≤3 mg/L)
- WD-02 : 완도 전복 가두리         — 26~27℃대 고수온(예비특보 수준), 용존산소 양호
- TY-03 : 통영 굴 수하식           — 23℃대, 정상 범위
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Any

# 데이터 기준 시각 (가상의 '현재'). 재현성을 위해 고정.
REFERENCE_TIME = datetime(2026, 8, 12, 9, 0)
HOURS = 96  # 최근 4일, 1시간 간격

FARMS: dict[str, dict[str, Any]] = {
    "GH-01": {
        "farm_id": "GH-01",
        "name": "고흥 나로 해상가두리 1호",
        "region": "전남 고흥",
        "sea_area": "고흥 득량만",
        "farm_type": "해상가두리",
        "species": "조피볼락",
        "species_upper_temp_c": 26.0,  # 사육 관리 상한(교육용 설정값)
        "stocking_density_kg_m3": 18.0,
        "cages": 24,
        "equipment": ["aerator", "liquid_oxygen", "feeder", "underwater_pump"],
        "contact": "가상 어가 A (연락처 비공개)",
    },
    "WD-02": {
        "farm_id": "WD-02",
        "name": "완도 노화 전복 가두리 2호",
        "region": "전남 완도",
        "sea_area": "완도 노화도 연안",
        "farm_type": "전복 가두리",
        "species": "참전복",
        "species_upper_temp_c": 27.0,
        "stocking_density_kg_m3": 9.5,
        "cages": 40,
        "equipment": ["aerator", "feeder", "shade_net"],
        "contact": "가상 어가 B (연락처 비공개)",
    },
    "TY-03": {
        "farm_id": "TY-03",
        "name": "통영 한산 굴 수하식 3호",
        "region": "경남 통영",
        "sea_area": "통영 한산만",
        "farm_type": "수하식",
        "species": "참굴",
        "species_upper_temp_c": 28.0,
        "stocking_density_kg_m3": 0.0,
        "cages": 0,
        "equipment": ["buoy_sensor"],
        "contact": "가상 어가 C (연락처 비공개)",
    },
}

# 시나리오별 파라미터: (수온 기저, 수온 상승추세/일, 일변동폭, DO 기저, DO 일변동폭, 염분)
_SCENARIO = {
    "GH-01": dict(t_base=27.6, t_trend=0.35, t_amp=0.9, do_base=4.3, do_amp=1.7, do_trend=-0.25, sal=31.2),
    "WD-02": dict(t_base=26.3, t_trend=0.25, t_amp=0.6, do_base=6.1, do_amp=0.8, do_trend=-0.05, sal=32.4),
    "TY-03": dict(t_base=23.1, t_trend=0.05, t_amp=0.4, do_base=7.2, do_amp=0.5, do_trend=0.0, sal=32.9),
}


def _series(farm_id: str) -> list[dict[str, Any]]:
    p = _SCENARIO[farm_id]
    rng = random.Random(f"aquasentinel-{farm_id}")
    rows = []
    start = REFERENCE_TIME - timedelta(hours=HOURS - 1)
    for i in range(HOURS):
        ts = start + timedelta(hours=i)
        day = i / 24.0
        # 수온: 오후 15시 최고, 새벽 최저
        temp = p["t_base"] + p["t_trend"] * day + p["t_amp"] * math.sin((ts.hour - 9) / 24 * 2 * math.pi)
        temp += rng.gauss(0, 0.08)
        # DO: 광합성 영향으로 오후 최고, 새벽 5시 전후 최저
        do = p["do_base"] + p["do_trend"] * day + p["do_amp"] * math.sin((ts.hour - 11) / 24 * 2 * math.pi)
        do += rng.gauss(0, 0.08)
        sal = p["sal"] + rng.gauss(0, 0.05)
        rows.append(
            {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                "water_temp_c": round(temp, 2),
                "do_mg_l": round(max(do, 0.3), 2),
                "salinity_psu": round(sal, 2),
            }
        )
    return rows


SENSOR_DB: dict[str, list[dict[str, Any]]] = {fid: _series(fid) for fid in FARMS}

# 기상 예보 (향후 3일, 가상)
WEATHER_FORECAST: dict[str, list[dict[str, Any]]] = {
    "GH-01": [
        {"date": "2026-08-12", "air_temp_max_c": 34.5, "wind_speed_ms": 1.8, "rain_mm": 0.0, "sky": "맑음"},
        {"date": "2026-08-13", "air_temp_max_c": 35.1, "wind_speed_ms": 1.5, "rain_mm": 0.0, "sky": "맑음"},
        {"date": "2026-08-14", "air_temp_max_c": 33.8, "wind_speed_ms": 2.6, "rain_mm": 2.0, "sky": "구름많음"},
    ],
    "WD-02": [
        {"date": "2026-08-12", "air_temp_max_c": 32.4, "wind_speed_ms": 3.4, "rain_mm": 0.0, "sky": "맑음"},
        {"date": "2026-08-13", "air_temp_max_c": 33.0, "wind_speed_ms": 2.9, "rain_mm": 0.0, "sky": "구름조금"},
        {"date": "2026-08-14", "air_temp_max_c": 31.6, "wind_speed_ms": 4.2, "rain_mm": 8.0, "sky": "비"},
    ],
    "TY-03": [
        {"date": "2026-08-12", "air_temp_max_c": 30.2, "wind_speed_ms": 4.8, "rain_mm": 0.0, "sky": "구름많음"},
        {"date": "2026-08-13", "air_temp_max_c": 29.5, "wind_speed_ms": 5.1, "rain_mm": 12.0, "sky": "비"},
        {"date": "2026-08-14", "air_temp_max_c": 29.0, "wind_speed_ms": 4.5, "rain_mm": 3.0, "sky": "흐림"},
    ],
}

# 기관 특보 (가상 속보). 실제 기관 발표가 아님.
OFFICIAL_ALERTS: dict[str, list[dict[str, Any]]] = {
    "고흥 득량만": [
        {"type": "고수온", "level": "경보", "issued_at": "2026-08-10 14:00", "issuer": "(가상) 수산과학 연구기관 속보"},
        {"type": "빈산소수괴", "level": "발생 속보", "issued_at": "2026-08-11 10:00", "issuer": "(가상) 수산과학 연구기관 속보"},
    ],
    "완도 노화도 연안": [
        {"type": "고수온", "level": "예비특보", "issued_at": "2026-08-09 11:00", "issuer": "(가상) 수산과학 연구기관 속보"},
    ],
    "통영 한산만": [],
}
