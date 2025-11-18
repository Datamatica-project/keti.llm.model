from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
from .storage import download_file
import tempfile

embedder = SentenceTransformer("dragonkue/snowflake-arctic-embed-l-v2.0-ko")

data = download_file("http://host.docker.internal:9000/", "minio", "miniostorage", "vectors", "index/faiss")

# FAISS 인덱스 로드 (바이너리 데이터를 메모리에서 읽기)
index_bytes = data.get("index")
if index_bytes:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(index_bytes)
        tmp_file.flush()
        faiss_index = faiss.read_index(tmp_file.name)
else:
    raise ValueError("인덱스 파일을 찾을 수 없습니다")

metadata = data.get("metadata", [])


def vector_search(query: str, top_k: int = 15) -> List[Dict]:
    # 쿼리 임베딩
    query_vec = embedder.encode([query])
    query_vec = normalize(query_vec, axis=1).astype("float32")

    # 검색 수행
    D, I = faiss_index.search(query_vec, top_k)

    results = []
    for i, score in zip(I[0], D[0]):
        if i < len(metadata) and i != -1:  # -1은 유효하지 않은 인덱스
            result_item = metadata[i].copy()  # 원본 수정 방지
            result_item.update({
                "similarity": float(1 / (1 + score)),
                "faiss_index": int(i),
                "raw_score": float(score)
            })
            results.append(result_item)

    return results