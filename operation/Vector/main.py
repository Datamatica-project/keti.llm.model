from config.load_storage import download_chunks_from_minio
from config.embedding import Embedder

def main():
    chunks = download_chunks_from_minio()

    if not chunks:
        print("임베딩할 chunk가 없습니다.")
        return

    embedder = Embedder()
    embedder.add_documents(chunks)           # 3. 임베딩
    embedder.save("C:/Users/dm_ohminchan/Model/operation/Vector/index/")                 # 4. 로컬 저장

    embedder.upload_to_minio(                # 5. MinIO 업로드
        bucket="vector",
        prefix="index/faiss/",
        endpoint_url="http://localhost:9000",
        access_key="minio",
        secret_key="miniostorage"
    )

if __name__ == "__main__":
    main()
