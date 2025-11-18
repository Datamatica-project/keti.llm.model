import boto3
import json


def download_file(endpoint_url: str, key: str, secret: str, bucket_name: str, prefix: str):
    s3 = boto3.client("s3",
                      endpoint_url=endpoint_url,
                      aws_access_key_id=key,
                      aws_secret_access_key=secret)

    response = s3.list_objects(Bucket=bucket_name, Prefix=prefix)
    loaded_data = {}

    if 'Contents' in response:
        for obj in response['Contents']:
            file_key = obj['Key']
            file_name = file_key.split('/')[-1]

            try:
                file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)

                if file_name == 'metadata.json' or 'metadata' in file_name:
                    content = file_obj['Body'].read().decode('utf-8')
                    loaded_data['metadata'] = json.loads(content)
                    print(f"metadata 로드 완료")

                elif 'index' in file_name:
                    content = file_obj['Body'].read()
                    loaded_data['index'] = content
                    print(f"index 로드 완료 ({len(content)} bytes)")

            except Exception as e:
                print(f"{file_name} 로드 실패: {e}")

    return loaded_data
