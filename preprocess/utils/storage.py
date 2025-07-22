import boto3
import json
from typing import List, Dict, Union, Optional
from botocore.exceptions import BotoCoreError, ClientError
from io import BytesIO

class StorageManager:
    def __init__(
        self,
        bucket: str,
        endpoint_url: str = "http://localhost:9000",
        aws_access_key_id: str = "minio",
        aws_secret_access_key: str = "miniostorage",
    ):
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

    def download(
        self,
        prefix: str = "",
        extensions: Union[str, List[str]] = ".json",
        verbose: bool = True
    ) -> List[Dict]:
        if isinstance(extensions, str):
            extensions = [extensions]

        try:
            response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            contents = response.get("Contents", [])
        except (BotoCoreError, ClientError) as e:
            print(f"목록 불러오기 실패: {e}")
            return []

        results = []
        for obj in contents:
            key = obj["Key"]
            if not any(key.endswith(ext) for ext in extensions):
                continue

            if verbose:
                print(f"Downloading: {key}")

            try:
                s3_obj = self.client.get_object(Bucket=self.bucket, Key=key)
                raw_data = s3_obj["Body"].read().decode("utf-8")
                json_data = json.loads(raw_data)
                results.append({
                    "key": key,
                    "raw": raw_data,
                    "content": json_data
                })
            except Exception as e:
                print(f"Failed to load {key}: {e}")

        if verbose:
            print(f"총 {len(results)}개 파일 다운로드 완료.")
        return results

    def upload_dataset(
            self,
            object_name: str,
            dataset,
            verbose: bool = True
    ) -> bool:
        try:
            buffer = BytesIO()
            dataset.to_parquet(buffer)
            buffer.seek(0)

            # boto3 클라이언트로 직접 업로드
            self.client.put_object(
                Bucket=self.bucket,
                Key=object_name,
                Body=buffer,
                ContentType="application/octet-stream"
            )

            if verbose:
                print(f"Dataset 업로드 성공: {object_name}")

            buffer.close()
            return True

        except Exception as e:
            print(f"Dataset 업로드 실패: {e}")
            return False