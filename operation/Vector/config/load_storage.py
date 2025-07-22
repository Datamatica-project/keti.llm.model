import boto3
import json
from typing import List, Dict, Union
from tqdm import tqdm

def download_chunks_from_minio(
    bucket: str = "chunk",
    prefix: str = "data/",
    endpoint_url: str = "http://localhost:9000",
    aws_access_key_id: str = "minio",
    aws_secret_access_key: str = "miniostorage",
    extensions: Union[str, List[str]] = ".json"
) -> List[Dict]:
    """
    MinIO에서 문서 chunk JSON을 다운로드하고, 각 chunk의 content와 메타정보를 추출합니다.
    """
    if isinstance(extensions, str):
        extensions = [extensions]

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = response.get("Contents", [])

    result = []
    for obj in tqdm(contents, desc="MinIO에서 파일 다운로드 중"):
        key = obj["Key"]
        if not any(key.endswith(ext) for ext in extensions):
            continue

        try:
            s3_object = s3.get_object(Bucket=bucket, Key=key)
            raw_data = s3_object["Body"].read().decode("utf-8")
            json_data = json.loads(raw_data)

            # 이중 리스트면 평탄화
            if isinstance(json_data, list) and json_data and isinstance(json_data[0], list):
                json_data = [chunk for sublist in json_data for chunk in sublist]

            for chunk in json_data:
                content = chunk.get("content", "").strip()
                if content:
                    result.append({
                        "title" : chunk.get("title"),
                        "text": content,
                        "document": chunk.get("document", "unknown"),
                        "chunk_id": chunk.get("chunk_id", ""),
                        "source": key
                    })

        except Exception as e:
            print(f"{key} 읽기 실패: {e}")

    print(f"총 {len(result)}개의 chunk 수집 완료.")
    return result
