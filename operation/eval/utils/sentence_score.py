from sentence_transformers import SentenceTransformer, util
from .models import similarity_model

def compute_semantic_similarity(candidate: str, reference: str) -> float:
    embeddings = similarity_model.encode([candidate, reference], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

