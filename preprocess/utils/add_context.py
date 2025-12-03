# -*- coding: utf-8 -*-
"""
add_context.py

- 기존 Q&A 데이터(QUESTION, ANSWER, source 등)를 읽어서
  각 항목마다 "context" 필드를 LLM으로 생성/추가한다.
- 결과는 새로운 JSON 파일로 저장한다.
- 429(Too Many Requests) 등에 대해서는 재시도(backoff)로 처리한다.
"""

from __future__ import annotations

from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
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

# 🔧 경로 (네 환경에 맞게 수정)
INPUT_PATH = Path(
    r"C:\Users\dm_ohminchan\RAGLLM-Feature-model-train\data\instrcution\qa_with_perspectives_cleaned.json"
)
OUTPUT_PATH = Path(
    r"C:\Users\dm_ohminchan\RAGLLM-Feature-model-train\data\instrcution\qa_with_context.json"
)

# 사용할 모델
DEFAULT_MODEL = "gpt-4.1-mini"

# 한 번에 전부 돌리기 부담되면 일부만 (None = 전체)
MAX_SAMPLES: int | None = None  # 예: 200 으로 두고 테스트 후 None 으로 바꾸기

# 이미 context가 있는 항목은 건너뛸지 여부
SKIP_IF_HAS_CONTEXT: bool = True

# ==========================
# 프롬프트
# ==========================

SYSTEM_PROMPT = """
당신은 한국어 농업 기술 보고서의 본문을 작성하는 전문가입니다.

당신의 임무:
- 주어진 질문(Q), 답변(A), 출처(source)를 기반으로
  이 Q&A를 뒷받침할 수 있는 "근거 문단(context)"을 작성하는 것이다.

반드시 다음 규칙을 지켜라.

1. 형식
- 출력은 **순수한 본문 텍스트 한 단락**만 포함해야 한다.
- JSON, 마크다운, 불릿포인트(-, *) 등은 절대 사용하지 않는다.
- "context:", "답:", "설명:" 같은 접두어를 붙이지 않는다.
- 인삿말, 메타 설명(예: "다음은 context입니다")은 쓰지 않는다.

2. 내용
- 실제 농업 기술/연구 보고서의 본문 일부처럼 자연스럽게 작성한다.
- 질문과 답변의 핵심 내용을 모두 포함하되, 이미 있는 문장을 그대로 복붙하지 말고
  보고서 문장처럼 자연스럽게 재구성한다.
- 3~6문장 정도의 하나의 단락으로 작성한다.
- 가능한 경우
  - 연구/사업의 목적, 대상 작물, 기술의 개요
  - 생육정보 측정 항목(생체정보, 환경정보 등)
  - 데이터 활용 방식(빅데이터 분석, 의사결정 지원 등)
  - 생산성 향상에 기여하는 논리적 근거
  등을 포함하면 좋다.
- 명확한 수치나 연도, 인명 등은 소설로 지어내지 말고, 모를 경우에는
  구체적인 수치 대신 일반적인 표현을 사용한다.

3. 출처 활용
- source에 포함된 과제명/보고서명을 참고하여, 해당 과제의
  목표나 성격을 자연스럽게 녹여낼 수 있다.
- 단, 실제 보고서를 읽지 않았으므로 너무 구체적인 사실(연도, 기관명 등)은 지양한다.
"""


def build_user_prompt(question: str, answer: str, source: str | None = None) -> str:
    base = f"""
다음은 Q&A와 출처 정보이다. 이 Q&A를 뒷받침할 수 있는 "근거 문단(context)"을 한 단락으로 작성하라.

Q: {question}
A: {answer}
"""
    if source:
        base += f"source: {source}\n"
    base += """
주의:
- 출력은 한 단락의 순수한 한국어 본문 텍스트만 작성한다.
- JSON, 마크다운, 불릿포인트, 따옴표 등은 사용하지 않는다.
- "context:" 같은 접두어를 붙이지 말고, 바로 문단 내용만 작성한다.
"""
    return base.strip()


# ==========================
# 데이터 구조
# ==========================


@dataclass
class QAItem:
    idx: int
    question: str
    answer: str
    source: str | None
    raw: Dict[str, Any]


# ==========================
# 유틸 함수들
# ==========================

def load_qa(path: Path) -> List[QAItem]:
    """기존 QA JSON 파일을 로드하여 QAItem 리스트로 변환."""
    if not path.exists():
        raise FileNotFoundError(f"입력 QA 파일을 찾을 수 없습니다: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw_list = json.load(f)

    items: List[QAItem] = []
    for idx, item in enumerate(raw_list):
        q = item.get("QUESTION") or item.get("question") or ""
        a = item.get("ANSWER") or item.get("answer") or ""
        s = item.get("source") or item.get("SOURCE") or item.get("source_document") or None

        if not q or not a:
            continue

        items.append(
            QAItem(
                idx=idx,
                question=q.strip(),
                answer=a.strip(),
                source=str(s).strip() if s else None,
                raw=item,
            )
        )

    logging.info(f"QA 로드 완료: 총 {len(items)}개 항목")
    return items


# 전역: 마지막 성공 호출 시각 (RPM 제한용)
LAST_CALL_TIME: float = 0.0
MIN_INTERVAL_SEC: float = 1.5  # 필요시 조정


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
            _wait_for_rate_limit()

            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.4,
                max_tokens=512,
                response_format={"type": "text"},
            )
            content = resp.choices[0].message.content.strip()
            LAST_CALL_TIME = time.time()
            return content

        except openai.RateLimitError as e:
            last_err = e
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


def save_results(path: Path, records: List[Dict[str, Any]]) -> None:
    """결과 리스트를 JSON 파일로 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logging.info(f"결과 저장 완료 → {path} (총 {len(records)}개 항목)")


# ==========================
# 메인 로직
# ==========================

def main():
    qa_items = load_qa(INPUT_PATH)

    # MAX_SAMPLES가 설정된 경우 일부만 사용
    if MAX_SAMPLES is not None:
        qa_items = qa_items[:MAX_SAMPLES]
        logging.info(f"MAX_SAMPLES={MAX_SAMPLES} → 앞에서 {len(qa_items)}개만 처리합니다.")

    augmented: List[Dict[str, Any]] = []
    total_done = 0

    for i, qa in enumerate(qa_items, start=1):
        raw = dict(qa.raw)  # 원본 복사

        # 이미 context가 있고, SKIP_IF_HAS_CONTEXT=True 이면 건너뜀
        if SKIP_IF_HAS_CONTEXT and "context" in raw and isinstance(raw["context"], str) and raw["context"].strip():
            logging.info(f"[{i}/{len(qa_items)}] 이미 context가 있어 건너뜀 (idx={qa.idx})")
            augmented.append(raw)
            total_done += 1
            continue

        logging.info(
            f"[{i}/{len(qa_items)}] context 생성 중... (idx={qa.idx}, source={qa.source})"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(qa.question, qa.answer, qa.source)},
        ]

        try:
            ctx = call_openai_with_retry(messages, model=DEFAULT_MODEL)
        except Exception as e:
            logging.error(f"[{i}] OpenAI 호출 실패, 이 항목은 context 없이 저장: {e}")
            # 실패해도 일단 원본은 보존
            augmented.append(raw)
            total_done += 1
            continue

        raw["context"] = ctx
        augmented.append(raw)
        total_done += 1

        # 주기적으로 중간 저장
        if total_done % 20 == 0:
            save_results(OUTPUT_PATH, augmented)

    # 최종 저장
    save_results(OUTPUT_PATH, augmented)
    logging.info("=== 전체 처리 완료 ===")


if __name__ == "__main__":
    main()
