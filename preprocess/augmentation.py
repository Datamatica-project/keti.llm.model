from utils.storage import StorageManager
from utils.generate import run_minio_qa_pipeline_v2
from datetime import datetime
import os


def main():
    storage = StorageManager("chunk")

    # 오늘 날짜
    today_str = datetime.today().strftime("%Y%m%d")

    # 저장 경로 구성
    output_dir = "/data/instruction/"
    os.makedirs(output_dir, exist_ok=True)  # 경로 없을 경우 생성
    output_file = os.path.join(output_dir, f"generation_QA_set_{today_str}.json")

    # 실행
    result = run_minio_qa_pipeline_v2(storage, prefix="data/", group_size=1, output_file=output_file)
    return result

if __name__ == "__main__":
    main()