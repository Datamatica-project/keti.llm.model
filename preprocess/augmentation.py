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