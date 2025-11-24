from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
from .storage import download_file
import tempfile

# ===== GPU 설정 =====
use_gpu = True  # GPU 강제 활성화 (False로 하면 CPU로 돌아감)
gpu_id = 0      # GPU 번호 (0번 GPU 사용)

# ===== 임베더 =====
embedder = SentenceTransformer(
    "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    device="cuda" if use_gpu else "cpu"
)

# ===== MinIO에서 벡터/메타 로드 =====
data = download_file(
    "http://host.docker.internal:9000/",
    "minio",
    "miniostorage",
    "vectors",
    "index/faiss"
)

# ===== FAISS 인덱스 로드 =====
index_bytes = data.get("index")
if not index_bytes:
    raise ValueError("인덱스 파일을 찾을 수 없습니다")

with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
    tmp_file.write(index_bytes)
    tmp_file.flush()

    # CPU 인덱스 읽기
    cpu_index = faiss.read_index(tmp_file.name)

# ===== GPU 변환 =====
if use_gpu:
    res = faiss.StandardGpuResources()               # GPU 리소스 준비
    faiss_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
else:
    faiss_index = cpu_index

# 메타데이터 로드
metadata = data.get("metadata", [])


# ===== 검색 함수 =====
def vector_search(query: str, top_k: int = 15) -> List[Dict]:
    # 쿼리 임베딩
    query_vec = embedder.encode([query])
    query_vec = normalize(query_vec, axis=1).astype("float32")

    # GPU에서 검색 수행
    D, I = faiss_index.search(query_vec, top_k)

    # 결과 정리
    results = []
    for i, score in zip(I[0], D[0]):
        if i < len(metadata) and i != -1:
            result_item = metadata[i].copy()
            result_item.update({
                "similarity": float(1 / (1 + score)),
                "faiss_index": int(i),
                "raw_score": float(score)
            })
            results.append(result_item)

    return results
