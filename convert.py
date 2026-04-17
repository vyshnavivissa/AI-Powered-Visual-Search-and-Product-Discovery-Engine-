import faiss
import numpy as np

index = faiss.read_index("faiss.index")
print(f"Total vectors: {index.ntotal}")

emb = index.reconstruct_n(0, index.ntotal)
print(f"Shape: {emb.shape}")

np.save("embeddings.npy", emb)
print("Done!")