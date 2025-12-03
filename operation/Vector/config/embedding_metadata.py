import faiss
import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "dragonkue/snowflake-arctic-embed-l-v2.0-ko", dim: int = 1024):
        self.model = SentenceTransformer(model_name)
        # L2 정규화 + Inner Product 기반 (코사인 유사도 방식)
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []
        self._global_counter = 0  # 전역 인덱스

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def add_documents(self, chunks: List[Dict]):
        """
        chunks: [{"text": "...", "chunk_id": "...", "document": "...", "date": "..."}, ...]
        metadata.json 구조 그대로 넣어도 됨.
        """
        for chunk in tqdm(chunks, desc="임베딩 및 인덱스에 추가 중"):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            # 벡터 임베딩 + 정규화
            vec = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False).astype("float32")
            vec = self._l2_normalize(vec)

            # 🔹 FAISS 인덱스에 추가
            self.index.add(vec)

            # 🔹 메타데이터 인덱싱
            global_index = self._global_counter
            self._global_counter += 1

            self.metadata.append({
                "index": global_index,
                "text": text,
                "chunk_id": chunk.get("chunk_id", ""),
                "document": chunk.get("document", ""),
                "date": chunk.get("date", "")
            })

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        # vector.index 저장
        faiss.write_index(self.index, os.path.join(save_dir, "vector.index"))

        # metadata.json 저장
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        print(f"저장 완료 → {save_dir}")


# ============================================================
# ✔ metadata.json을 읽어서 인덱스 생성하는 함수
# ============================================================
def build_index_from_metadata(
    metadata_path: str,
    save_dir: str,
    model_name: str = "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    dim: int = 1024,
) -> Embedder:
    """
    metadata.json을 그대로 읽어서 text 필드를 임베딩하여
    → vector.index 생성
    → metadata.json 그대로 저장
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        chunks: List[Dict] = json.load(f)

    embedder = Embedder(model_name=model_name, dim=dim)
    embedder.add_documents(chunks)
    embedder.save(save_dir)

    return embedder


# ============================================================
# 실행 예시
# ============================================================
if __name__ == "__main__":
    METADATA_PATH = "C:/Users/dm_ohminchan/RAGLLM-Feature-model-train/operation/Vector/index/metadata.json"
    SAVE_DIR = "C:/Users/dm_ohminchan/RAGLLM-Feature-model-train/operation/Vector/index/"

    # metadata.json → vector.index 생성
    build_index_from_metadata(
        metadata_path=METADATA_PATH,
        save_dir=SAVE_DIR,
    )

