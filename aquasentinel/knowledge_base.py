"""양식장 위기대응 지식베이스 + 경량 BM25 검색기 (외부 벡터DB 없이 동작하는 RAG).

- 한국어 형태소 분석기 없이도 동작하도록 '문자 2-gram + 공백 토큰' 혼합 토크나이저 사용
- 각 청크에는 출처(source)를 붙여, 에이전트가 [KB-xx] 형태로 인용하도록 강제한다
- 내용은 공개 보도자료/기사에서 확인한 기준과 일반적 관리요령을 교육용으로 요약한 것이며,
  'source' 가 '교육용 정리'인 항목은 현장 표준 매뉴얼이 아님
"""

from __future__ import annotations

import math
import re
from collections import Counter

KB_CHUNKS: list[dict[str, str]] = [
    {"id": "KB-01", "title": "고수온 예비특보 기준",
     "text": "해역 수온이 25℃에 도달했거나 도달할 것으로 예상되는 경우 고수온 예비특보가 발령된다.",
     "source": "세계일보 2026-07-14 보도(해양수산부 고수온 대응)"},
    {"id": "KB-02", "title": "고수온 주의보·경보 기준",
     "text": "고수온 주의보는 수온 28℃에 도달했거나 도달할 것으로 예상될 때, 고수온 경보는 28℃에 도달해 3일 이상 지속되거나 지속될 것으로 예상될 때 발령된다.",
     "source": "트윅24 2026-07-30 보도(전남 6개 해역 고수온 경보)"},
    {"id": "KB-03", "title": "고수온 시 관리요령(정부 지도 사항)",
     "text": "고수온 폐사 방지를 위해 조기 출하 및 긴급 방류, 고수온 대응 장비 총동원, 사육 밀도 조정, 사료 공급 중단 등의 관리요령을 지도한다. 사료 공급을 줄이거나 중단하고 대응장비를 가동한다.",
     "source": "트윅24 2026-07-30 보도(해수부 관리요령)"},
    {"id": "KB-04", "title": "빈산소수괴 정의",
     "text": "빈산소수괴(산소부족 물덩어리)는 바닷물에 녹아 있는 산소 농도가 3mg/L 이하인 물덩어리로, 어패류의 호흡을 방해해 양식생물 피해를 일으킬 수 있다.",
     "source": "노컷뉴스 보도(국립수산과학원 빈산소수괴 자료집)"},
    {"id": "KB-05", "title": "빈산소 발생 시 산소공급 조치",
     "text": "용존산소가 빈산소 기준 이하로 떨어지는 새벽 시간대에는 산소발생기·액화산소 공급장치를 우선 가동하고, 가동 중인 산소 공급 장비를 임의로 정지하지 않는다. 장비 가동 후 1시간 간격으로 용존산소를 재측정한다.",
     "source": "교육용 정리(일반 관리요령 요약, 현장 표준 매뉴얼 아님)"},
    {"id": "KB-06", "title": "고수온·빈산소 중첩 시 급이 관리",
     "text": "수온이 높고 용존산소가 낮은 시기에는 먹이 소화 과정에서 산소 소비가 늘어나므로 급이를 중단하거나 최소화한다. 급이 재개는 수온과 용존산소가 회복된 뒤 소량부터 시작한다.",
     "source": "교육용 정리(일반 관리요령 요약, 현장 표준 매뉴얼 아님)"},
    {"id": "KB-07", "title": "취약 시간대 집중 감시",
     "text": "용존산소는 광합성이 멈춘 야간부터 새벽까지 낮아지고 오후에 높아지는 일주기를 보인다. 새벽 시간대 용존산소, 오후 최고수온, 무풍일 여부를 핵심 모니터링 지표로 삼는다.",
     "source": "교육용 정리(일반 관리요령 요약, 현장 표준 매뉴얼 아님)"},
    {"id": "KB-08", "title": "보고 및 피해 신고",
     "text": "이상 폐사가 발생하면 폐사체를 신속히 수거하고 관할 지자체와 수산 연구기관에 피해 상황을 신고한다. 신고 시 발생 일시, 수온, 용존산소, 폐사량을 함께 기록한다.",
     "source": "교육용 정리(일반 관리요령 요약, 현장 표준 매뉴얼 아님)"},
    {"id": "KB-09", "title": "정상 단계 예방 관리",
     "text": "위기 신호가 없는 정상 단계에서는 정기 센서 점검, 대응 장비 시운전, 사육 밀도 점검으로 예비특보 이전에 대비한다.",
     "source": "교육용 정리(일반 관리요령 요약, 현장 표준 매뉴얼 아님)"},
    {"id": "KB-10", "title": "차광·수심 조정",
     "text": "고수온기에는 차광막을 설치하거나 가두리·수하연의 수심을 조정해 표층 고수온 노출을 줄이는 방법을 검토한다.",
     "source": "교육용 정리(일반 관리요령 요약, 현장 표준 매뉴얼 아님)"},
]


def _tokens(text: str) -> list[str]:
    text = re.sub(r"[^\w가-힣\.]", " ", text.lower())
    words = text.split()
    grams = []
    for w in words:
        grams.append(w)
        grams.extend(w[i:i + 2] for i in range(len(w) - 1))
    return grams


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [_tokens(d) for d in docs]
        self.avgdl = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def score(self, query: str) -> list[float]:
        q = _tokens(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


_INDEX = BM25([c["title"] + " " + c["text"] for c in KB_CHUNKS])


def search(query: str, top_k: int = 3) -> list[dict[str, object]]:
    scores = _INDEX.score(query)
    ranked = sorted(range(len(KB_CHUNKS)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [dict(KB_CHUNKS[i], score=round(scores[i], 3)) for i in ranked if scores[i] > 0]


def get_ids() -> set[str]:
    return {c["id"] for c in KB_CHUNKS}
