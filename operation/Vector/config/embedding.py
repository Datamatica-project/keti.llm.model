import faiss
import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import boto3


class Embedder:
    def __init__(self, model_name: str = "dragonkue/snowflake-arctic-embed-l-v2.0-ko", dim: int = 1024):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatL2(dim)  # Cosine similarity는 normalize 필요
        self.metadata = []


    def add_documents(self, chunks: List[Dict]):
        for chunk in tqdm(chunks, desc="임베딩 및 인덱스에 추가 중"):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            # 벡터 임베딩
            vector = self.model.encode([text])  # shape: (1, dim)
            self.index.add(vector)  # FAISS에 추가

            self.metadata.append({
                "title" : chunk.get("title", ""),
                "text" : text,
                "chunk_id": chunk.get("chunk_id", ""),
                "document": chunk.get("document", ""),
                "source": chunk.get("source", "")
            })

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        # FAISS index 저장
        faiss.write_index(self.index, os.path.join(save_dir, "vector.index"))

        # 메타데이터 저장
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        print(f"벡터 인덱스 및 메타데이터 저장 완료 → {save_dir}")


    def upload_to_minio(
            self,
            bucket: str,
            prefix: str,
            endpoint_url: str,
            access_key: str,
            secret_key: str,
            directory: str = "C:/Users/dm_ohminchan/Model/operation/Vector/index/"  # 🔥 default directory
    ):
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

        for file_name in ["vector.index", "metadata.json"]:
            file_path = os.path.join(directory, file_name)  # 디렉토리 경로 포함
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일이 존재하지 않음: {file_path}")

            s3.upload_file(file_path, bucket, f"{prefix.rstrip('/')}/{file_name}")
            print(f"벡터 데이터 업로드 완료: {file_name} → {bucket}/{prefix}")