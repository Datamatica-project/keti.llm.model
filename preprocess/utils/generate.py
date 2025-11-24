# -*- coding: utf-8 -*-
"""
generate_qa_with_perspectives.py

- 벡터 인덱스용 metadata.json을 읽어서
  각 청크(text)마다 의미 판단 + Q&A 생성
- 결과는 qa_with_perspectives.json 으로 저장
- 429(Too Many Requests) 등에 대해서는 재시도(backoff)로 처리
"""

from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import os
import json
import time
import random
from dotenv import load_dotenv
import openai
from openai import OpenAI
load_dotenv()

# ==========================
# 기본 설정
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment or .env file")

client = OpenAI()

# 경로 (네 환경에 맞게 수정 가능)
METADATA_PATH = Path(r"C:\Users\dm_ohminchan\RAGLLM-Feature-model-train\operation\Vector\index\metadata.json")
OUTPUT_PATH = Path(r"C:\Users\dm_ohminchan\RAGLLM-Feature-model-train\data\instrcution\qa_with_perspectives.json")

# 모델
DEFAULT_MODEL = "gpt-4.1-mini"

# 사용할 관점 라벨 정의 (검증용)
PERSPECTIVE_LABELS = {
    "기초지식",
    "실무응용",
    "문제해결",
    "비교분석",
    "친환경농법",
}

# ==========================
# 프롬프트
# ==========================

SYSTEM_PROMPT = """
너는 한국어 농업 Q&A 데이터셋을 생성하는 JSON 생성기이다.

반드시 아래 규칙을 지켜야 한다.

1. 출력 형식
- 항상 **순수 JSON만** 출력해야 한다.
- JSON 이외의 설명, 주석, 코드블록, 자연어 문장을 절대 추가하지 않는다.
- 최상위 구조는 리스트(List)이다.
    - 예: [] 또는 [{ ... }, { ... }]

2. 의미 판단 규칙
- 사용자가 제공한 '청크 텍스트'를 읽고, 농업 관련 Q&A를 만들 가치가 있는지 먼저 판단한다.
- 청크가 다음과 같으면 **즉시 빈 배열([])만 출력**한다.
    - 목차, 페이지 번호, 표 제목, 장/절 제목만 있는 경우
    - 동일 문장/표현이 거의 반복되는 경우
    - "참고문헌", "부록", "서론", "요약" 등 실제 기술 내용이 거의 없는 경우
    - 농업과 거의 무관한 행정/서식/일반 설명 등인 경우
    - 참고문헌 목록, 인용문(저자, 연도, 논문 제목, 학술지, 권/호, 페이지, DOI, URL 등)만 있는 경우
    - 웹사이트 주소(https://...), 파일명(.bmp, .jpg, .png 등), 픽셀 크기 등 이미지/파일 메타데이터만 있는 경우
- 의미 없다고 판단하면 Q&A를 만들지 말고 **그냥 [] 하나만 출력**한다.

3. Q&A 생성 규칙
- 의미가 있다고 판단되는 경우에만 Q&A를 생성한다.
- 하나의 청크에서 1~3개의 Q&A만 생성한다. (너무 많이 만들지 말 것)
- 각 항목은 아래와 같은 JSON 객체여야 한다:

{
  "QUESTION": "<사용자가 실제로 물어볼 법한 자연스러운 질문>",
  "ANSWER": "<청크 내용을 바탕으로 한 구체적이고 실무적인 한국어 답변>",
  "PERSPECTIVE": "<관점 라벨>"
}

4. QUESTION 작성 규칙
- 실제 농업인이 물어볼 것 같은 구체적인 질문으로 작성한다.
- "무엇인가요?" 보다는 "언제", "어떻게", "어떤 조건에서" 등을 명시한다.
- 한 문장으로 자연스럽게 작성한다.

5. ANSWER 작성 규칙
- 반드시 청크에 근거하여 작성하고 헛소리를 지어내지 않는다.
- 3~8문장 정도로, 너무 짧지도 길지도 않게 작성한다.
- 가능한 한 **작업 시기, 조건, 수치, 단계별 절차** 등을 포함한다.
- 문장체로 자연스럽게 작성하되, 인삿말·감사 표현은 절대 넣지 않는다.

6. PERSPECTIVE(관점 라벨) 규칙
- 아래 다섯 개 라벨 중 하나만 사용해야 한다. 다른 값은 절대 사용하지 않는다.
    - "기초지식"    : 기본 개념, 용어 정의, 원리 설명 중심
    - "실무응용"    : 현장에서 바로 쓸 수 있는 작업 요령, 관리 방법, 노하우 중심
    - "문제해결"    : 병해충, 생리장해, 실패 원인 분석, 해결책 중심
    - "비교분석"    : 품종, 재배 방법, 자재, 기술 간 장단점 비교 중심
    - "친환경농법"  : 유기농, 저투입, 친환경 자재, 환경 보호 관점 중심
- 반드시 문자열 하나로만 채워야 하며, **빈 문자열("")을 절대 사용하지 않는다.**
- 위 라벨 중 어느 것에 넣을지 애매할 때는 **"실무응용"**을 사용한다.

7. 기타
- 각 Q&A는 독립적인 데이터이므로, 서로를 언급하지 않는다.
- JSON 포맷 오류(따옴표 누락, 쉼표 오류 등)를 절대 내지 않는다.
- null, true, false 같은 값은 사용하지 말고, 모두 문자열로 처리한다.
"""


def build_user_prompt(chunk_text: str) -> str:
    return f"""
다음은 농업 관련 문서의 한 청크(일부 내용)이다. 이 텍스트를 읽고 위 지침에 따라
1) 의미 없는 경우에는 **빈 배열([])** 만 출력하고,
2) 의미가 있다면 1~3개의 Q&A를 생성하여 JSON 리스트로 출력하라.

[청크 텍스트 시작]
{chunk_text}
[청크 텍스트 끝]

주의:
- 반드시 **유효한 JSON만** 출력할 것
- 최상위는 리스트여야 한다. (예: [] 또는 [{{...}}, {{...}}])
- JSON 외의 다른 텍스트는 절대 출력하지 말 것
"""


# ==========================
# 데이터 구조
# ==========================

@dataclass
class Chunk:
    idx: int
    text: str
    document: str | None = None
    chunk_id: str | None = None


# ==========================
# 유틸 함수들
# ==========================

def load_metadata(path: Path) -> List[Chunk]:
    """벡터 인덱스 metadata.json 로드 → Chunk 리스트로 변환"""
    if not path.exists():
        raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    chunks: List[Chunk] = []
    for i, item in enumerate(raw):
        text = item.get("text") or item.get("content") or ""
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                idx=i,
                text=text.strip(),
                document=item.get("document") or item.get("source") or "",
                chunk_id=str(item.get("chunk_id") or ""),
            )
        )

    logging.info(f"메타데이터 로드 완료: 총 {len(chunks)}개 청크")
    return chunks


def normalize_perspective(label: Any) -> str:
    """모델이 준 PERSPECTIVE 값을 허용 라벨 중 하나로 정규화."""
    s = ""
    if isinstance(label, str):
        s = label.strip()
    if s in PERSPECTIVE_LABELS:
        return s

    logging.warning(f"알 수 없는 관점 라벨 발견: {s!r} → '실무응용'으로 대체")
    return "실무응용"


# 전역: 마지막 성공 호출 시각 (RPM 제한용)
LAST_CALL_TIME: float = 0.0
# 고정 20초 딜레이는 제거하고, 실제 429가 발생할 때만 backoff 하도록 함
MIN_INTERVAL_SEC: float = 2.0  # 필요하면 1.0 정도로 바꿔서 완충 가능


def _wait_for_rate_limit():
    """마지막 호출 시점 기준으로 최소 MIN_INTERVAL_SEC 만큼 간격을 맞춘다."""
    global LAST_CALL_TIME
    now = time.time()
    elapsed = now - LAST_CALL_TIME
    if elapsed < MIN_INTERVAL_SEC:
        sleep_sec = MIN_INTERVAL_SEC - elapsed
        logging.info(f"레이트 리밋 보호를 위해 {sleep_sec:.2f}초 대기...")
        time.sleep(sleep_sec)


def call_openai_with_retry(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    max_retries: int = 5,
) -> str:
    """
    MIN_INTERVAL_SEC로 호출 간격을 조절하고,
    429/500 계열 에러에 대해 재시도하는 OpenAI 호출 래퍼.
    """
    global LAST_CALL_TIME
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            # 전역 타이머 기준으로 최소 간격 보장
            _wait_for_rate_limit()

            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "text"},  # 텍스트(JSON 문자열)로 받음
            )
            content = resp.choices[0].message.content.strip()

            # 성공 시점 기록 (다음 요청과의 간격 계산용)
            LAST_CALL_TIME = time.time()

            return content

        except openai.RateLimitError as e:
            last_err = e
            # ➜ 추가 backoff (429일 때는 서버가 힘들다는 뜻이므로 더 쉰다)
            sleep_sec = 10.0 + random.random() * 5.0
            logging.warning(
                f"RateLimitError 발생 (시도 {attempt+1}/{max_retries}) → {sleep_sec:.2f}초 대기 후 재시도: {e}"
            )
            time.sleep(sleep_sec)

        except openai.APIError as e:
            last_err = e
            sleep_sec = 5.0 + random.random() * 3.0
            logging.warning(
                f"APIError 발생 (시도 {attempt+1}/{max_retries}) → {sleep_sec:.2f}초 대기 후 재시도: {e}"
            )
            time.sleep(sleep_sec)

        except Exception as e:
            last_err = e
            logging.error(f"예상치 못한 오류 발생: {e}", exc_info=True)
            break

    raise RuntimeError(f"OpenAI 호출 실패 (재시도 {max_retries}회): {last_err}")


def parse_model_output(raw_content: str) -> List[Dict[str, str]]:
    """모델이 반환한 JSON 문자열을 파싱하고, 필드/라벨을 정리."""
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        logging.error(
            f"JSON 파싱 실패 → 이 청크는 건너뜀: {e}\n내용 일부: {raw_content[:200]!r}"
        )
        return []

    if not isinstance(data, list):
        logging.warning(f"최상위가 리스트가 아님 → 무시: 타입={type(data)}")
        return []

    results: List[Dict[str, str]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        q = str(item.get("QUESTION", "")).strip()
        a = str(item.get("ANSWER", "")).strip()
        if not q or not a:
            continue

        p = normalize_perspective(item.get("PERSPECTIVE", ""))

        results.append(
            {
                "QUESTION": q,
                "ANSWER": a,
                "PERSPECTIVE": p,
            }
        )

    return results


def save_results(path: Path, records: List[Dict[str, Any]]) -> None:
    """결과 리스트를 JSON 파일로 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logging.info(f"결과 저장 완료 → {path} (총 {len(records)}개 Q&A)")


# ==========================
# 메인 로직
# ==========================

def main():
    chunks = load_metadata(METADATA_PATH)

    all_qa: List[Dict[str, Any]] = []
    total_valid = 0

    for idx, ch in enumerate(chunks, start=1):
        logging.info(
            f"[{idx}/{len(chunks)}] 청크 처리 중... "
            f"source={ch.document} / model={DEFAULT_MODEL}"
        )

        # 1) 프롬프트 구성
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(ch.text)},
        ]

        # 2) OpenAI 호출 (레이트 리밋-aware + 재시도)
        try:
            raw = call_openai_with_retry(messages, model=DEFAULT_MODEL)
        except Exception as e:
            logging.error(f"[{idx}] OpenAI 호출 실패, 이 청크는 건너뜀: {e}")
            continue

        # 3) JSON 파싱 + 관점 정리
        qa_list = parse_model_output(raw)
        if not qa_list:
            continue

        # 4) 원본 메타데이터 정보 붙이기
        for qa in qa_list:
            qa["source_document"] = ch.document
            qa["chunk_index"] = ch.idx
            qa["chunk_id"] = ch.chunk_id

        all_qa.extend(qa_list)
        total_valid += len(qa_list)
        logging.info(f"    유효 Q&A 생성: {len(qa_list)}개 (누적 {total_valid})")

        # 5) 주기적으로 중간 저장 (예: 10개 단위)
        if total_valid % 10 == 0:
            save_results(OUTPUT_PATH, all_qa)

    # 최종 저장
    save_results(OUTPUT_PATH, all_qa)
    logging.info("=== 전체 처리 완료 ===")


if __name__ == "__main__":
    main()
