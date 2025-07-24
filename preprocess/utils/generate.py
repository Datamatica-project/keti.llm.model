from openai import OpenAI
import logging
import os
import json
import time
from dotenv import load_dotenv

# 기본 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# OpenAI 키 설정
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = OpenAI()


def generate_enhanced_qa(context, domain="농업", model="gpt-4.1-nano-2025-04-14", num_questions=25):
    """개선된 조건으로 고품질 QA 생성"""
    prompt_template = f"""
다음은 한국어로 작성된 전문 농업 기술 문서의 일부입니다. 이 문서를 바탕으로 질문을 생성하세요.
---------------------
{context}
---------------------

당신은 '{domain}' 분야의 현장 경험이 풍부한 농업 전문가입니다. 아래 조건에 맞춰 **정확히 {num_questions}개의 질문과 답변 쌍**을 JSON 형식으로 출력하세요.

조건:
1. **답변은 반드시 700자 이상**
2. **구체적 수치, 방법, 시기 포함**
3. **실무에서 바로 적용 가능한 내용**
4. **다음 구조로 답변 작성:**
   - 전문가 도입부: "농업 전문가로서 상세히 답변드리겠습니다."
   - 핵심 내용 (문서 기반)
   - **구체적 방법**: 단계별 실행 방안
   - **실무 팁**: 현장 노하우
   - **주의사항**: 실제 주의점

추가 조건:
- 모든 질문은 문서 내용만 바탕으로 하세요. 외부 지식 사용 금지.
- 문서의 특정 문장을 기반으로 하되, 연락처/출판 정보는 질문 금지.
- 문서 내 농업 기술, 재배 방법, 품종, 기계 사용법, 저장 유통 기술 등 실질적 내용만 사용.
- 반드시 존댓말 사용.

출력 형식 (JSON Only, 리스트 없이 객체만 콤마로 구분):
{{
  "QUESTION": "구체적이고 실용적인 질문...",
  "ANSWER": "\\n\\n[핵심 내용 700자 이상]\\n\\n**구체적 방법:**\\n- 단계별 실행 방안들...\\n\\n**실무 팁:**\\n- 현장에서 검증된 노하우들...\\n\\n**주의사항:**\\n- 실제 농장에서 주의해야 할 점들..."
}},
{{"QUESTION": "...", "ANSWER": "..."}}, ...
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_template}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f'GPT 호출 중 오류 발생: {e}')
        return None


def custom_json_parser_safe(response):
    """안전한 JSON 파싱"""
    try:
        json_string = response.content.strip() if hasattr(response, 'content') else response.strip()
        json_string = json_string.removeprefix("```json\n").removesuffix("\n```").strip()
        json_string = f'[{json_string}]'.replace(',]', ']')
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logging.warning(f"JSON 파싱 실패, 수동 파싱 시도: {e}")
        return []
    except Exception as e:
        logging.error(f"파싱 중 오류: {e}")
        return []


def generate_templates_batch(context, domain, model="gpt-4.1-nano-2025-04-14",
                             total_questions=1000, batch_size=25):
    """배치 방식으로 QA 생성"""
    all_questions = []

    for i in range(0, total_questions, batch_size):
        remaining = min(batch_size, total_questions - i)
        batch_num = i // batch_size + 1
        total_batches = (total_questions + batch_size - 1) // batch_size

        logging.info(f"배치 {batch_num}/{total_batches}: {remaining}개 질문 생성 중...")

        batch_result = generate_enhanced_qa(context, domain, model, remaining)

        if batch_result:
            parsed_batch = custom_json_parser_safe(batch_result)
            all_questions.extend(parsed_batch)
            logging.info(f"배치 {batch_num} 성공: {len(parsed_batch)}개 질문")
        else:
            logging.warning(f"배치 {batch_num} 생성 실패")

        if batch_num < total_batches:
            time.sleep(5)

    return all_questions


def group_titles_for_qa(documents, group_size=3):
    """MinIO 문서들을 title별로 그룹화"""
    grouped_docs = []

    for doc in documents:
        content = doc.get("content", [])
        title_data = {}

        for chunk in content:
            title = chunk.get('title', '제목없음')
            text = chunk.get('content', '')
            if text.strip():
                title_data[title] = text

        titles = list(title_data.keys())
        for i in range(0, len(titles), group_size):
            group_titles = titles[i:i + group_size]
            merged_content = "\n\n".join([f"## {title}\n{title_data[title]}" for title in group_titles])
            grouped_docs.append({
                "content": merged_content,
                "source": doc['key'],
                "titles": group_titles
            })

    return grouped_docs


def run_minio_qa_pipeline_v2(storage_manager, prefix="", group_size=3, output_file="qa_result.json"):
    """관점 확장 + 배치 기반 QA 생성 파이프라인"""
    documents = storage_manager.download(prefix=prefix, extensions=".json")
    logging.info(f"문서 {len(documents)}개 다운로드")

    grouped_docs = group_titles_for_qa(documents, group_size)
    logging.info(f"그룹 {len(grouped_docs)}개 생성")

    perspectives = [
        ("기초지식", "기본적인 농업 기술과 원리"),
        ("실무응용", "현장에서 직접 적용하는 방법"),
        ("문제해결", "문제 상황 발생 시 대처법"),
        ("비교분석", "다른 방법과의 비교 및 장단점"),
        ("친환경농법", "유기농/저투입 기술 중심 관점"),
    ]

    all_qa = []
    for i, doc in enumerate(grouped_docs, 1):
        logging.info(f"[{i}/{len(grouped_docs)}] 그룹 QA 증강 시작: {doc['titles']}")

        for perspective_name, perspective_desc in perspectives:
            enhanced_context = f"""
            관점: {perspective_name} - {perspective_desc}
            {doc['content']}
            """
            batch_qa = generate_templates_batch(
                context=enhanced_context,
                domain="농업",
                model="gpt-4.1-nano-2025-04-14",
                total_questions=40,
                batch_size=5
            )

            for qa in batch_qa:
                qa["source"] = doc["source"]
                qa["perspective"] = perspective_name

            all_qa.extend(batch_qa)
            logging.info(f"  관점 {perspective_name}: {len(batch_qa)}개 생성됨")

            for idx, qa in enumerate(batch_qa[:2], 1):
                logging.info(f"    ▶ 샘플 QA {idx}")
                logging.info(f"      Q: {qa['QUESTION']}")
                logging.info(f"      A: {qa['ANSWER'][:200]}...")

        logging.info(f"→ 그룹 누적: {len(all_qa)}개 QA")

        if i % 10 == 0:
            logging.info(f"==== 진행 상황 {i}/{len(grouped_docs)} ====")
            logging.info(f"누적 QA 수: {len(all_qa)}개")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

    logging.info(f"\n최종 완료: {len(all_qa)}개 QA 저장됨 → {output_file}")
    return all_qa
