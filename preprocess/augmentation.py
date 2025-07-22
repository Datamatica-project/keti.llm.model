<<<<<<< HEAD
from openai import OpenAI
import json
import time
import os

# OpenAI 클라이언트 설정
client = OpenAI(
    api_key='api')


def generate_quality_qa_batch(text, num_questions=25):
    """배치용 고품질 QA 생성 (최대 30개까지)"""

    prompt = f"""
다음 농업 문서에서 {num_questions}개의 전문적인 질문과 답변을 만드세요.

문서:
{text}

조건:
1. 답변은 반드시 500자 이상
2. 구체적 수치, 방법, 시기 포함
3. 실무에서 바로 적용 가능한 내용
4. 다음 구조로 답변 작성:
   - 전문가 도입부: "농업 전문가로서 상세히 답변드리겠습니다."
   - 핵심 내용 (문서 기반)
   - **구체적 방법**: 단계별 실행 방안
   - **실무 팁**: 현장 노하우
   - **주의사항**: 실제 주의점

JSON 배열 형식으로만 출력:
[
  {{
    "QUESTION": "구체적이고 실용적인 질문",
    "ANSWER": "농업 전문가로서 상세히 답변드리겠습니다.\\n\\n[핵심 내용]\\n\\n**구체적 방법:**\\n- 방법들\\n\\n**실무 팁:**\\n- 팁들\\n\\n**주의사항:**\\n- 주의점들"
  }}
]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ GPT 호출 오류: {e}")
        return None


def parse_json_safe(raw_response):
    """안전한 JSON 파싱"""
    try:
        # 마크다운 제거
        clean_response = raw_response.replace("```json", "").replace("```", "").strip()
        qa_list = json.loads(clean_response)
        return qa_list
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 실패: {e}")
        return None


def batch_generate_qa(text, total_questions=1000, batch_size=25, save_file="batch_qa_results.json"):
    """배치 처리로 대량 QA 생성"""

    print("🚀 배치 처리 QA 생성 시작")
    print(f"🎯 목표: {total_questions}개")
    print(f"📦 배치 크기: {batch_size}개")
    print(f"📄 텍스트 길이: {len(text)}자")
    print("=" * 60)

    # 배치 계산
    num_batches = (total_questions + batch_size - 1) // batch_size
    print(f"📊 총 {num_batches}번의 배치 호출 예정")

    all_qa = []
    success_count = 0
    fail_count = 0

    for batch_num in range(num_batches):
        # 남은 질문 수 계산
        remaining = min(batch_size, total_questions - len(all_qa))

        print(f"\n🔄 배치 {batch_num + 1}/{num_batches}")
        print(f"   생성할 QA: {remaining}개")
        print(f"   누적 생성: {len(all_qa)}개")

        # 배치 실행
        start_time = time.time()
        raw_response = generate_quality_qa_batch(text, remaining)
        end_time = time.time()

        if raw_response:
            # JSON 파싱
            qa_batch = parse_json_safe(raw_response)

            if qa_batch:
                # 품질 검증
                valid_qa = []
                for qa in qa_batch:
                    if validate_qa_quality(qa):
                        valid_qa.append(qa)

                all_qa.extend(valid_qa)
                success_count += 1

                print(f"   ✅ 성공: {len(valid_qa)}개 생성")
                print(f"   ⏱️ 소요시간: {end_time - start_time:.1f}초")

                # 중간 저장 (50개마다)
                if len(all_qa) % 50 == 0:
                    save_intermediate(all_qa, f"temp_qa_{len(all_qa)}.json")
            else:
                fail_count += 1
                print(f"   ❌ 파싱 실패")
        else:
            fail_count += 1
            print(f"   ❌ GPT 호출 실패")

        # 진행률 표시
        progress = (batch_num + 1) / num_batches * 100
        print(f"   📈 진행률: {progress:.1f}%")

        # API 제한 방지 대기
        if batch_num < num_batches - 1:
            wait_time = 3
            print(f"   ⏳ {wait_time}초 대기...")
            time.sleep(wait_time)

    # 최종 결과
    print(f"\n🎉 배치 생성 완료!")
    print(f"📊 최종 통계:")
    print(f"   목표: {total_questions}개")
    print(f"   실제 생성: {len(all_qa)}개")
    print(f"   성공률: {len(all_qa) / total_questions * 100:.1f}%")
    print(f"   성공 배치: {success_count}/{num_batches}")
    print(f"   실패 배치: {fail_count}/{num_batches}")

    # 품질 통계
    if all_qa:
        avg_length = sum(len(qa['ANSWER']) for qa in all_qa) / len(all_qa)
        long_enough = sum(1 for qa in all_qa if len(qa['ANSWER']) >= 500)

        print(f"\n📊 품질 통계:")
        print(f"   평균 답변 길이: {avg_length:.0f}자")
        print(f"   500자 이상: {long_enough}/{len(all_qa)}개 ({long_enough / len(all_qa) * 100:.1f}%)")

    # 최종 저장
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

    print(f"💾 {save_file}에 저장 완료")

    return all_qa


def validate_qa_quality(qa):
    """QA 품질 검증"""
    question = qa.get('QUESTION', '')
    answer = qa.get('ANSWER', '')

    # 기본 검증
    if not question or not answer:
        return False

    # 길이 검증
    if len(answer) < 500:
        return False

    # 구조 검증
    if '농업 전문가' not in answer:
        return False

    # 섹션 검증
    required_sections = ['구체적', '방법', '팁', '주의']
    section_count = sum(1 for section in required_sections if section in answer)

    if section_count < 3:
        return False

    return True


def save_intermediate(qa_list, filename):
    """중간 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)
    print(f"   💾 중간 저장: {filename}")


def chunk_and_generate(long_text, total_questions=1000, chunk_size=5000):
    """긴 텍스트를 청크로 나누어 QA 생성"""

    print("✂️ 텍스트 청크 분할 + QA 생성")
    print(f"📄 전체 텍스트: {len(long_text)}자")
    print(f"📏 청크 크기: {chunk_size}자")

    # 텍스트 청크 분할
    chunks = []
    for i in range(0, len(long_text), chunk_size):
        chunk = long_text[i:i + chunk_size]
        if len(chunk) > 1000:  # 너무 짧은 청크 제외
            chunks.append(chunk)

    print(f"📊 총 {len(chunks)}개 청크 생성")

    # 청크당 QA 수 계산
    qa_per_chunk = total_questions // len(chunks)
    remaining_qa = total_questions % len(chunks)

    print(f"🎯 청크당 {qa_per_chunk}개 QA (마지막 청크 +{remaining_qa}개)")

    all_qa = []

    for i, chunk in enumerate(chunks):
        chunk_qa_count = qa_per_chunk + (remaining_qa if i == len(chunks) - 1 else 0)

        print(f"\n📝 청크 {i + 1}/{len(chunks)} 처리")
        print(f"   길이: {len(chunk)}자")
        print(f"   목표 QA: {chunk_qa_count}개")

        # 청크별 배치 생성
        chunk_qa = batch_generate_qa(
            chunk,
            total_questions=chunk_qa_count,
            batch_size=25,
            save_file=f"chunk_{i + 1}_qa.json"
        )

        all_qa.extend(chunk_qa)
        print(f"   ✅ 청크 {i + 1} 완료: {len(chunk_qa)}개")

    # 전체 결과 저장
    final_file = "final_all_chunks_qa.json"
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 전체 청크 처리 완료!")
    print(f"📊 총 {len(all_qa)}개 QA 생성")
    print(f"💾 {final_file}에 최종 저장")

    return all_qa


def test_batch_system():
    """배치 시스템 테스트"""

    sample_text = """
    토마토 재배에서 물 관리는 매우 중요한 요소입니다. 토마토는 뿌리가 비교적 얕게 분포하므로 
    표토층의 수분 관리에 특별한 주의가 필요합니다. 과습할 경우 뿌리썩음병, 역병 등의 
    토양전염성 병해가 발생하기 쉽고, 반대로 건조할 경우에는 열과(과실 터짐) 현상이 
    나타날 수 있습니다.

    생육 단계별로 보면, 정식 후 활착기에는 토양 수분을 60-70% 수준으로 유지하고, 
    개화착과기에는 적당한 수분 스트레스를 주어 착과를 촉진시키며, 과실 비대기에는 
    충분한 수분을 공급하여 과실의 크기와 품질을 향상시켜야 합니다.

    관수 시기는 오전 8-10시경이 가장 적절하며, 이때 관수하면 하루 종일 충분한 
    증산작용을 통해 양분 흡수가 원활해집니다. 점적관수 시스템을 활용하면 
    물의 이용효율을 높이고 병해 발생을 줄일 수 있습니다.

    시설재배에서는 환기와 온도 관리가 병행되어야 합니다. 과습한 환경에서는 
    잿빛곰팡이병, 역병 등이 발생하기 쉬우므로 적절한 환기를 통해 습도를 조절해야 합니다.
    온도는 주간 25-28°C, 야간 15-18°C로 유지하는 것이 좋습니다.
    """ * 10  # 텍스트 확장

    print("🧪 배치 시스템 테스트 (50개 QA)")

    # 테스트: 50개 QA 생성
    result = batch_generate_qa(
        text=sample_text,
        total_questions=50,
        batch_size=25,
        save_file="test_batch_50qa.json"
    )

    return result


def main():
    """메인 실행"""
    print("🚀 배치 처리 기반 대량 QA 생성 시스템")
    print("=" * 60)

    while True:
        print("\n📋 메뉴:")
        print("1. 테스트 (50개 QA)")
        print("2. 소량 생성 (100개 QA)")
        print("3. 중량 생성 (500개 QA)")
        print("4. 대량 생성 (1000개 QA)")
        print("5. 청크 분할 생성 (긴 텍스트용)")
        print("6. 종료")

        choice = input("\n선택하세요 (1-6): ")

        if choice == "1":
            test_batch_system()

        elif choice == "2":
            text = input("텍스트를 입력하세요 (또는 파일 경로): ")
            if os.path.exists(text):
                with open(text, 'r', encoding='utf-8') as f:
                    text = f.read()
            batch_generate_qa(text, total_questions=100, save_file="qa_100.json")

        elif choice == "3":
            text = input("텍스트를 입력하세요 (또는 파일 경로): ")
            if os.path.exists(text):
                with open(text, 'r', encoding='utf-8') as f:
                    text = f.read()
            batch_generate_qa(text, total_questions=500, save_file="qa_500.json")

        elif choice == "4":
            text = input("텍스트를 입력하세요 (또는 파일 경로): ")
            if os.path.exists(text):
                with open(text, 'r', encoding='utf-8') as f:
                    text = f.read()
            batch_generate_qa(text, total_questions=1000, save_file="qa_1000.json")

        elif choice == "5":
            text = input("긴 텍스트 파일 경로를 입력하세요: ")
            if os.path.exists(text):
                with open(text, 'r', encoding='utf-8') as f:
                    long_text = f.read()
                total = int(input("총 생성할 QA 수를 입력하세요: "))
                chunk_and_generate(long_text, total_questions=total)
            else:
                print("❌ 파일을 찾을 수 없습니다.")

        elif choice == "6":
            print("👋 프로그램을 종료합니다.")
            break

        else:
            print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()
=======
import logging
from collections import defaultdict
from utils.storage import StorageManager
from utils.generate import (
    generate_templates_batch, get_questions_config,
    enhance_qa_dataset, filter_quality_qa, format_for_instruction_tuning
)
import os
import json

SAVE_DIR = "C:/Users/dm_ohminchan/Model/data/instrcution/"


def main():
    result = StorageManager("chunk").download()
    document_groups = defaultdict(list)

    for file in result:
        contents = file["content"]

        # 이중 리스트 평탄화
        if isinstance(contents, list) and contents and isinstance(contents[0], list):
            contents = [chunk for sublist in contents for chunk in sublist]

        for chunk in contents:
            doc_name = chunk.get("document")
            if doc_name:
                document_groups[doc_name].append(chunk)

    # 문서별로 content 합치기
    merged_documents = []
    for doc_name, chunks in document_groups.items():
        # content를 순서대로 연결
        merged_text = "\n".join(chunk.get("content", "") for chunk in chunks)

        merged_documents.append({
            "document": doc_name,
            "merged_content": merged_text,
        })

    return merged_documents


def generate_all():
    documents = main()

    for doc in documents[6:9]:
        context = doc["merged_content"]
        doc_name = doc["document"]
        domain = "농업"

        total_q, batch_size = get_questions_config(len(context))
        logging.info(f"문서: {doc_name} | 길이: {len(context)}자 → 질문 {total_q}개 생성")

        # 1. 원본 QA 생성
        qa_list = generate_templates_batch(
            context, domain,
            total_questions=total_q,
            batch_size=batch_size
        )

        if not qa_list:
            logging.warning(f"{doc_name} 질문 생성 실패 — 생략")
            continue

        logging.info(f"{doc_name} 원본 QA {len(qa_list)}개 생성 완료")

        # 2. 데이터셋 개선
        logging.info(f"{doc_name} 데이터셋 개선 시작...")
        enhanced_data = enhance_qa_dataset(qa_list)
        logging.info(f"{doc_name} 개선 완료: {len(enhanced_data)}개")

        # 3. 품질 필터링
        quality_data = filter_quality_qa(enhanced_data, min_answer_length=200)
        logging.info(f"{doc_name} 품질 필터링 후: {len(quality_data)}개")

        # 4. Instruction tuning 형태로 포맷
        final_data = format_for_instruction_tuning(quality_data)
        logging.info(f"{doc_name} 최종 데이터: {len(final_data)}개")

        # 5. 파일 저장 (3가지 버전)
        base_filename = doc_name.replace('.pdf', '')

        # 원본 데이터 저장
        original_file = os.path.join(SAVE_DIR, f"original_{base_filename}.json")
        with open(original_file, "w", encoding="utf-8") as f:
            json.dump(qa_list, f, ensure_ascii=False, indent=2)

        # 개선된 데이터 저장 (chat template 형태)
        enhanced_file = os.path.join(SAVE_DIR, f"enhanced_{base_filename}.json")
        with open(enhanced_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        # 단순 QA 형태로도 저장 (호환성)
        simple_qa = [{"QUESTION": item["QUESTION"], "ANSWER": item["ANSWER"]} for item in final_data]
        simple_file = os.path.join(SAVE_DIR, f"qa_{base_filename}.json")
        with open(simple_file, "w", encoding="utf-8") as f:
            json.dump(simple_qa, f, ensure_ascii=False, indent=2)

        logging.info(f"{doc_name} 저장 완료:")
        logging.info(f"  - 원본: {original_file}")
        logging.info(f"  - 개선: {enhanced_file}")
        logging.info(f"  - 호환: {simple_file}")

        # 개선 결과 샘플 출력
        if final_data:
            logging.info(f"\n=== {doc_name} 개선 결과 샘플 ===")
            sample = final_data[0]
            logging.info(f"질문: {sample['QUESTION']}")
            logging.info(f"답변: {sample['ANSWER'][:200]}...")


def generate_enhanced_only():
    """개선된 데이터만 생성하는 버전 (더 간단)"""
    documents = main()

    for doc in documents[6:9]:
        context = doc["merged_content"]
        doc_name = doc["document"]
        domain = "농업"

        total_q, batch_size = get_questions_config(len(context))
        logging.info(f"문서: {doc_name} | 길이: {len(context)}자 → 질문 {total_q}개 생성")

        # QA 생성 및 즉시 개선
        qa_list = generate_templates_batch(
            context, domain,
            total_questions=total_q,
            batch_size=batch_size
        )

        if not qa_list:
            logging.warning(f"{doc_name} 질문 생성 실패 — 생략")
            continue

        # 원스텝 개선 처리
        enhanced_data = enhance_qa_dataset(qa_list)
        quality_data = filter_quality_qa(enhanced_data)
        final_data = format_for_instruction_tuning(quality_data)

        # 최종 파일만 저장
        filename = os.path.join(SAVE_DIR, f"enhanced_{doc_name.replace('.pdf', '')}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        logging.info(f"{doc_name}: 원본 {len(qa_list)}개 → 최종 {len(final_data)}개 저장 완료")


def enhance_existing_files():
    """이미 생성된 원본 파일들을 개선하는 함수"""
    for filename in os.listdir(SAVE_DIR):
        if filename.startswith("qa_") and filename.endswith(".json") and not filename.startswith("enhanced_"):
            filepath = os.path.join(SAVE_DIR, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                original_data = json.load(f)

            logging.info(f"{filename} 개선 시작: {len(original_data)}개")

            # 개선 처리
            enhanced_data = enhance_qa_dataset(original_data)
            quality_data = filter_quality_qa(enhanced_data)
            final_data = format_for_instruction_tuning(quality_data)

            # 개선된 파일 저장
            enhanced_filename = filename.replace("qa_", "enhanced_")
            enhanced_filepath = os.path.join(SAVE_DIR, enhanced_filename)

            with open(enhanced_filepath, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)

            logging.info(f"{filename}: {len(original_data)}개 → {len(final_data)}개로 개선 완료")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 옵션 1: 전체 프로세스 (원본 + 개선된 버전 모두 저장)
    generate_all()

    # 옵션 2: 개선된 버전만 생성 (더 간단)
    # generate_enhanced_only()

    # 옵션 3: 기존 파일들을 개선 (이미 원본이 있을 때)
    enhance_existing_files()
>>>>>>> 0c89ddde410f1f3f2d72ee223ac04c56dff24b2e
