import faiss
import json
import numpy as np

index = faiss.read_index("vector.index")


print("총 벡터 개수:", index.ntotal)
print("차원 수:", index.d)

if index.ntotal > 0:
    vec = np.zeros((1, index.d), dtype='float32')
    for i in range(10):
        index.reconstruct(i, vec[0])
    print("첫 번째 벡터:", vec[0][:10])
