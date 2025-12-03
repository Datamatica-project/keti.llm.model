from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
from .storage import download_file
import tempfile

# ===== 임베더: GPU 사용 =====
use_gpu_for_embedder = True

embedder = SentenceTransformer(
    "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    device="cuda" if use_gpu_for_embedder else "cpu"
)

# ===== MinIO에서 벡터/메타 로드 =====
data = download_file(
    "http://host.docker.internal:9000/",
    "minio",
    "miniostorage",
    "vectors",
    "index/faiss"
)

# ===== FAISS 인덱스 로드 (항상 CPU) =====
index_bytes = data.get("index")
if not index_bytes:
    raise ValueError("인덱스 파일을 찾을 수 없습니다")

with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
    tmp_file.write(index_bytes)
    tmp_file.flush()

    faiss_index = faiss.read_index(tmp_file.name)

# 메타데이터 로드
metadata = data.get("metadata", [])


# ===== 검색 함수 =====
def vector_search(query: str, top_k: int = 15) -> List[Dict]:
    # 쿼리 임베딩은 GPU에서 수행 (위에서 device="cuda")
    query_vec = embedder.encode([query])
    query_vec = normalize(query_vec, axis=1).astype("float32")

    # 🔍 검색은 CPU FAISS 인덱스로 수행
    D, I = faiss_index.search(query_vec, top_k)

    # 결과 정리
    results = []
    for i, score in zip(I[0], D[0]):
        if i < len(metadata) and i != -1:
            result_item = metadata[i].copy()
            result_item.update({
                "similarity": float(1 / (1 + score)),
                "faiss_index": int(i),
                "raw_score": float(score),
            })
            results.append(result_item)

    return results
