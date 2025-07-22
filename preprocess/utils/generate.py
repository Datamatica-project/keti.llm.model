from openai import OpenAI
import logging
import os
import json
import time
import random
import re

os.environ["OPENAI_API_KEY"]= "API"
client = OpenAI()


def generate_templates(context, domain, model="gpt-4.1-nano-2025-04-14", num_questions=100):
    prompt_template = f"""
다음은 한국어로 작성된 전문 농업 기술 문서의 일부입니다. 이 문서를 바탕으로 질문을 생성하세요.
---------------------
{context}
---------------------

당신은 '{domain}' 분야의 대학교수입니다. 아래 조건에 맞춰 **정확히 {num_questions}개의 질문과 답변 쌍**을 JSON 형식으로 출력하세요.

조건:
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
  "QUESTION": "...",
  "ANSWER": "..."
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
    """
    안전한 JSON 파싱
    """
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


def get_questions_config(text_length: int) -> tuple[int, int]:
    """
    텍스트 길이에 따라 질문 수와 배치 크기 결정
    """

    config_map = [
        (100000, (1500, 50)),
        (80000, (1200, 50)),
        (60000, (1000, 50)),
        (40000, (700, 40)),
        (20000, (500, 40)),
        (10000, (300, 30)),
        (5000, (200, 25)),
        (2000, (100, 20)),
        (0, (50, 10)),
    ]

    return next((q for threshold, q in config_map if text_length >= threshold), (50, 10))


def generate_templates_batch(context, domain, model="gpt-4.1-nano-2025-04-14",
                             total_questions=100, batch_size=25):
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
        batch_result = generate_templates(context, domain, model, remaining)

        if batch_result:
            try:
                parsed_batch = custom_json_parser_safe(batch_result)
                all_questions.extend(parsed_batch)
                print(f"배치 {batch_num} 성공: {len(parsed_batch)}개 질문")
            except Exception as e:
                print(f"배치 {batch_num} 파싱 실패: {e}")
                continue
        else:
            print(f"  ❌ 배치 {batch_num} 생성 실패")

        # 배치 간 잠시 대기 (API 레이트 리밋 방지)
        if batch_num < total_batches:  # 마지막 배치가 아닐 때만 대기
            time.sleep(2)

    return all_questions



def analyze_question_type(question):
    """질문 유형 분석"""
    if any(word in question for word in ['재배', '키우', '심기', '파종', '작물']):
        return 'crop_cultivation'
    elif any(word in question for word in ['축사', '사육', '가축', '닭', '소', '돼지']):
        return 'livestock'
    elif any(word in question for word in ['기계', '장비', '트랙터', '관리기']):
        return 'machinery'
    else:
        return 'general'


def get_enhancement_intro():
    """다양한 인트로 문구"""
    intros = [
        "농업 전문가로서 상세히 답변드리겠습니다.",
        "실제 농장 경험을 바탕으로 설명하겠습니다.",
        "농업 현장에서 검증된 방법을 안내해드리겠습니다.",
        "전문적인 농업 기술 관점에서 답변드리겠습니다."
    ]
    return random.choice(intros)


def add_detailed_sections(question, question_type):
    """구체적 방법 섹션 추가"""
    if '온도' in question or '환경' in question:
        return """**구체적 관리 방법:**
- 온도: 적정 범위 유지 및 계절별 조절 방법
- 습도: 환기 시설을 통한 습도 조절
- 환경 모니터링: 정기적인 측정 및 기록 관리"""

    elif '방제' in question or '병해충' in question:
        return """**통합적 방제 전략:**
- 예방적 방제: 환경 관리 및 저항성 품종 활용
- 생물학적 방제: 천적 곤충 및 미생물 활용
- 화학적 방제: 안전사용기준 준수한 농약 살포
- 물리적 방제: 끈끈이 트랩, 방충망 등 활용"""

    elif '재배' in question or '키우' in question:
        return """**체계적 재배 관리:**
- 토양 준비: pH 조절 및 유기물 투입
- 품종 선택: 지역 적응성 및 시장성 고려
- 재배 일정: 파종/정식 시기 및 생육 단계별 관리
- 수확 및 저장: 적기 수확 및 품질 유지 방법"""

    else:
        return """**전문적 접근 방법:**
- 과학적 근거에 바탕한 체계적 관리
- 경제성과 효율성을 고려한 실용적 방법
- 지속가능한 농업을 위한 환경친화적 접근"""


def add_practical_tips(question_type):
    """실무 팁 추가"""
    tips_by_type = {
        'crop_cultivation': [
            "토양 검정을 통한 과학적 시비로 비료 효율성 극대화",
            "생육 단계별 관찰 일지 작성으로 문제 조기 발견",
            "기상 정보 활용한 농작업 일정 최적화"
        ],
        'livestock': [
            "동물 행동 관찰을 통한 건강 상태 조기 진단",
            "사료 효율 개선을 위한 급여 시간 및 량 조절",
            "스트레스 최소화를 위한 환경 개선"
        ],
        'machinery': [
            "정기 점검으로 고장 예방 및 수명 연장",
            "작업 조건에 맞는 적정 속도 및 깊이 조절",
            "안전 수칙 준수로 사고 예방"
        ],
        'general': [
            "체계적 기록 관리로 농장 운영 데이터 축적",
            "전문가 자문 및 교육 참여로 기술 향상",
            "경영 분석을 통한 수익성 개선"
        ]
    }

    selected_tips = tips_by_type.get(question_type, tips_by_type['general'])
    tip_text = "\n- ".join([""] + random.sample(selected_tips, min(2, len(selected_tips))))

    return f"**실무 팁:**{tip_text}"


def add_precautions(question_type):
    """주의사항 추가"""
    precautions_by_type = {
        'crop_cultivation': [
            "과도한 물 공급으로 인한 뿌리 썩음 주의",
            "농약 사용 시 안전사용기준 및 수확전 사용금지기간 준수",
            "연작 피해 방지를 위한 윤작 계획 수립"
        ],
        'livestock': [
            "급격한 사료 변경으로 인한 소화 장애 방지",
            "질병 전파 차단을 위한 차단 방역 철저",
            "과밀 사육으로 인한 스트레스 및 질병 발생 주의"
        ],
        'machinery': [
            "작업 전 안전 점검으로 사고 예방",
            "과부하 작업으로 인한 기계 손상 방지",
            "정비 불량으로 인한 작업 효율 저하 주의"
        ],
        'general': [
            "무분별한 정보 적용보다는 전문가 상담 우선",
            "단기 이익보다는 장기적 지속가능성 고려",
            "안전 수칙 준수로 농작업 사고 예방"
        ]
    }

    selected_precautions = precautions_by_type.get(question_type, precautions_by_type['general'])
    precaution_text = "\n- ".join([""] + random.sample(selected_precautions, min(2, len(selected_precautions))))

    return f"**주의사항:**{precaution_text}"


def enhance_answer_with_expertise(question, original_answer):
    """전문성을 강화한 답변으로 확장"""
    question_type = analyze_question_type(question)

    # 인트로 선택
    intro = get_enhancement_intro()

    # 원본 답변 정리
    cleaned_answer = original_answer.strip()
    if not cleaned_answer.endswith('.'):
        cleaned_answer += '.'

    # 구조화된 답변 생성
    enhanced_answer = f"""{intro}

{cleaned_answer}

{add_detailed_sections(question, question_type)}

{add_practical_tips(question_type)}

{add_precautions(question_type)}"""

    return enhanced_answer


def diversify_questions(question):
    """질문을 더 자연스럽고 다양하게 변형"""
    question_variations = {
        r'어떤.*조치': ['어떻게 관리해야', '무엇을 주의해야', '어떤 방법이 효과적'],
        r'어떤.*환경': ['어떤 조건이 좋은지', '최적의 환경은 무엇인지', '이상적인 조건'],
        r'.*방법.*무엇': ['구체적인 방법', '실제 적용 방안', '효과적인 접근법'],
        r'.*때.*어떻게': ['시기별 관리법', '상황별 대처법', '단계별 방법']
    }

    for pattern, variations in question_variations.items():
        if re.search(pattern, question):
            variation = random.choice(variations)
            return re.sub(pattern, variation, question)

    return question


def enhance_qa_dataset(qa_data):
    """전체 QA 데이터셋 개선"""
    enhanced_dataset = []

    for i, qa in enumerate(qa_data):
        try:
            # 질문 다양화
            enhanced_question = diversify_questions(qa['QUESTION'])

            # 답변 전문성 강화
            enhanced_answer = enhance_answer_with_expertise(
                enhanced_question,
                qa['ANSWER']
            )

            enhanced_qa = {
                'QUESTION': enhanced_question,
                'ANSWER': enhanced_answer
            }

            enhanced_dataset.append(enhanced_qa)

            if (i + 1) % 100 == 0:
                print(f"진행 상황: {i + 1}개 완료")

        except Exception as e:
            print(f"QA {i} 처리 중 오류: {e}")
            continue

    return enhanced_dataset


def filter_quality_qa(qa_data, min_answer_length=200):
    """품질 기준으로 QA 필터링"""
    quality_qa = []

    quality_keywords = ['구체적', '방법', '단계', '주의', '관리', '적정', '권장', '예방']

    for qa in qa_data:
        answer = qa['ANSWER']

        # 기본 길이 체크
        if len(answer) < min_answer_length:
            continue

        # 품질 키워드 체크
        keyword_count = sum(1 for keyword in quality_keywords if keyword in answer)
        if keyword_count < 3:
            continue

        # 구조화 체크 (** 섹션이 있는지)
        if answer.count('**') < 4:
            continue

        quality_qa.append(qa)

    return quality_qa


def format_for_instruction_tuning(qa_data):
    """Instruction tuning 형태로 포맷팅"""
    formatted_data = []

    for qa in qa_data:
        # Chat template 형태로 구성
        messages = [
            {
                "role": "system",
                "content": "당신은 농업 분야의 전문가입니다. 농민들에게 실용적이고 과학적인 농업 기술을 제공하며, 현장에서 바로 적용할 수 있는 구체적인 조언을 해주세요."
            },
            {
                "role": "user",
                "content": qa['QUESTION']
            },
            {
                "role": "assistant",
                "content": qa['ANSWER']
            }
        ]

        formatted_data.append({
            'messages': messages,
            'QUESTION': qa['QUESTION'],
            'ANSWER': qa['ANSWER']
        })

    return formatted_data


def enhance_existing_dataset(input_file, output_file):
    """기존 데이터셋을 개선하여 저장"""

    # 데이터 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    print(f"원본 데이터: {len(original_data)}개")

    # 데이터 개선
    enhanced_data = enhance_qa_dataset(original_data)
    print(f"개선 완료: {len(enhanced_data)}개")

    # 품질 필터링
    quality_data = filter_quality_qa(enhanced_data)
    print(f"품질 필터링 후: {len(quality_data)}개")

    # Instruction tuning 형태로 포맷
    final_data = format_for_instruction_tuning(quality_data)
    print(f"최종 데이터: {len(final_data)}개")

    # 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"개선된 데이터셋이 {output_file}에 저장되었습니다.")

    # 샘플 출력
    if final_data:
        print("\n=== 개선 결과 샘플 ===")
        sample = final_data[0]
        print(f"질문: {sample['QUESTION']}")
        print(f"답변: {sample['ANSWER'][:300]}...")


# 사용 예시
if __name__ == "__main__":
    # 기존 QA 데이터 개선
    # enhance_existing_dataset('original_qa.json', 'enhanced_agriculture_qa.json')

    # 또는 메모리에서 직접 처리
    # enhanced_data = enhance_qa_dataset(your_qa_list)
    # quality_data = filter_quality_qa(enhanced_data)
    # final_data = format_for_instruction_tuning(quality_data)
    pass