from openai import OpenAI
import logging
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_API_KEY")
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
- 모든 질문은 상기 문서의 내용만 바탕으로 생성해야 하며, 외부 지식을 포함해서는 안 됩니다.
- 질문은 반드시 문서 내의 특정 문장을 기반으로 출제되어야 하며, 근거 문장이 없는 경우 질문을 만들지 마세요.
- 문서에 명시된 조건, 방법, 시기, 위치, 예시, 구체적 사례 등에 기반해 질문을 작성하세요.
- 질문은 반드시 해당 문서의 농업 기술, 재배 방법, 품종 정보, 기계 사용법, 유통·저장 기술 등 실질적인 농업 지식만을 대상으로 해야 합니다.
- 반말이 아닌 존댓말을 사용하여야 합니다.
- 아래 항목에 관련된 내용은 질문/답변으로 생성하지 마세요:
  - 집필자, 감수자, 연구원 이름
  - 전화번호, 이메일 등 연락처
  - 목차, 부록, 출판 정보, 문서 제목 또는 장 제목만 나열된 구문

출력 형식 (JSON Only, 리스트 없이 개별 오브젝트만 콤마로 구분):
{{
  "QUESTION": "구체적이고 실용적인 질문...",
  "ANSWER": "\\n\\n[핵심 내용 700자 이상]\\n\\n**구체적 방법:**\\n- 단계별 실행 방안들...\\n\\n**실무 팁:**\\n- 현장에서 검증된 노하우들...\\n\\n**주의사항:**\\n- 실제 농장에서 주의해야 할 점들..."
}},
{{
  "QUESTION": "...",
  "ANSWER": "..."
}},
...

출력에는 어떠한 마크다운이나 설명도 포함하지 마세요. JSON 오브젝트만 출력하세요.
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
        if hasattr(response, 'content'):
            json_string = response.content.strip()
        else:
            json_string = response.strip()

        # 마크다운 제거
        json_string = json_string.removeprefix("```json\n").removesuffix("\n```").strip()

        # 배열 형태로 변환
        json_string = f'[{json_string}]'

        # 마지막 쉼표 제거
        json_string = json_string.replace(',]', ']')

        return json.loads(json_string)

    except json.JSONDecodeError as e:
        print(f"JSON 파싱 실패, 수동 파싱 시도: {e}")
        return []

    except Exception as e:
        print(f"파싱 중 오류: {e}")
        return []


def generate_templates_batch(context, domain, model="gpt-4.1-nano-2025-04-14",
                             total_questions=1000, batch_size=25):
    """
    배치 방식으로 질문 생성
    """
    all_questions = []

    for i in range(0, total_questions, batch_size):
        remaining = min(batch_size, total_questions - i)
        batch_num = i // batch_size + 1
        total_batches = (total_questions + batch_size - 1) // batch_size

        print(f"배치 {batch_num}/{total_batches}: {remaining}개 질문 생성 중...")

        # 기존 generate_templates 함수 호출
        batch_result = generate_enhanced_qa(context, domain, model, remaining)

        if batch_result:
            try:
                parsed_batch = custom_json_parser_safe(batch_result)
                all_questions.extend(parsed_batch)
                print(f"배치 {batch_num} 성공: {len(parsed_batch)}개 질문")
            except Exception as e:
                print(f"배치 {batch_num} 파싱 실패: {e}")
                continue
        else:
            print(f"  배치 {batch_num} 생성 실패")

        # 배치 간 잠시 대기 (API 레이트 리밋 방지)
        if batch_num < total_batches:  # 마지막 배치가 아닐 때만 대기
            time.sleep(5)

    return all_questions


def group_titles_for_qa(documents, group_size=3):
    """MinIO 문서들을 title별로 그룹화"""
    grouped_docs = []

    for doc in documents:
        content = doc.get("content", [])

        # title별 내용 수집
        title_data = {}
        for chunk in content:
            title = chunk.get('title', '제목없음')
            text = chunk.get('content', '')
            if text.strip():
                title_data[title] = text

        # group_size개씩 묶기
        titles = list(title_data.keys())
        for i in range(0, len(titles), group_size):
            group_titles = titles[i:i + group_size]

            # 내용 병합
            merged_content = "\n\n".join([
                f"## {title}\n{title_data[title]}"
                for title in group_titles
            ])

            grouped_docs.append({
                "content": merged_content,
                "source": doc['key'],
                "titles": group_titles
            })

    return grouped_docs


def run_minio_qa_pipeline(storage_manager, prefix="", group_size=3, output_file="qa_result.json"):
    """MinIO → 그룹화 → QA생성 → 저장"""

    # 1. MinIO에서 다운로드
    documents = storage_manager.download(prefix=prefix, extensions=".json")
    print(f"문서 {len(documents)}개 다운로드")

    # 2. title별 그룹화
    grouped_docs = group_titles_for_qa(documents, group_size)
    print(f"그룹 {len(grouped_docs)}개 생성")

    # 3. 각 그룹별 다각도 QA 증강
    perspectives = [
        ("기초지식", "기본적인 농업 기술과 원리"),
        ("실무응용", "현장에서 직접 적용하는 방법"),
        ("문제해결", "문제 상황 발생 시 대처법"),
        ("비교분석", "다른 방법과의 비교 및 장단점")
    ]

    all_qa = []
    for i, doc in enumerate(grouped_docs, 1):
        print(f"\n[{i}/{len(grouped_docs)}] 그룹 QA 증강 시작")
        print(f"제목들: {', '.join(doc['titles'][:3])}{'...' if len(doc['titles']) > 3 else ''}")
        print(f"내용 길이: {len(doc['content'])}자")

        # 질문 개수 결정
        text_len = len(doc['content'])
        base_num_q = 3 if text_len < 2000 else 5 if text_len < 4000 else 7
        print(f"각 관점별 목표 질문 수: {base_num_q}개")

        group_qa_count = 0

        # 각 관점별로 QA 생성
        for j, (perspective_name, perspective_desc) in enumerate(perspectives, 1):
            print(f"  [{j}/4] {perspective_name} 관점 QA 생성 중...")

            enhanced_context = f"""
            관점: {perspective_name} - {perspective_desc}
            
            {doc['content']}
            """

            qa_result = generate_enhanced_qa(enhanced_context, "농업", "gpt-4.1-nano-2025-04-14", base_num_q)

            if qa_result:
                parsed = custom_json_parser_safe(qa_result)
                for qa in parsed:
                    qa['source'] = doc['source']
                    qa['perspective'] = perspective_name
                all_qa.extend(parsed)
                group_qa_count += len(parsed)
                print(f"    → {len(parsed)}개 QA 생성됨")

                # 생성된 QA 내용 출력
                for idx, qa in enumerate(parsed, 1):
                    print(f"    QA {idx}:")
                    print(f"      Q: {qa['QUESTION']}")
                    print(
                        f"      A: {qa['ANSWER']}..." if len(qa['ANSWER']) > 100 else f"      A: {qa['ANSWER']}")
                    print()
            else:
                print(f"    → 생성 실패")

        print(f"그룹 총 QA: {group_qa_count}개 | 전체 누적: {len(all_qa)}개")

        # 10개 그룹마다 진행상황 요약
        if i % 10 == 0:
            print(f"\n=== 진행 상황 ({i}/{len(grouped_docs)}) ===")
            print(f"현재까지 총 QA: {len(all_qa)}개")
            print(f"예상 최종 QA: {len(all_qa) * len(grouped_docs) // i}개")
            print("=" * 40)

    # 4. 결과 저장

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

    print(f"완료! 총 {len(all_qa)}개 QA")
    return all_qa
