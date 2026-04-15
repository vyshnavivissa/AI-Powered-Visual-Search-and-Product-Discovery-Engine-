import os
import numpy as np
import faiss
from PIL import Image

from backend.model import get_image_embedding

IMAGE_FOLDER = r"C:\Users\xx\OneDrive\Desktop\visual_search\data\fashion"

if not os.path.exists(IMAGE_FOLDER):
    raise FileNotFoundError(f"Image folder '{IMAGE_FOLDER}' not found.")
os.makedirs("models", exist_ok=True)

embeddings = []
image_paths = []

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}  

for file in os.listdir(IMAGE_FOLDER):
    if not any(file.lower().endswith(ext) for ext in VALID_EXTENSIONS):
        print(f"Skipped (not an image): {file}")
        continue

    path = os.path.join(IMAGE_FOLDER, file)

    try:
        image = Image.open(path).convert("RGB")
        emb = get_image_embedding(image)
        embeddings.append(emb[0])
        image_paths.append(file)
        print(f"Processed {file}")

    except Exception as e:
        print(f"Error {file}: {e}")

if not embeddings:
    raise ValueError("No valid embeddings generated. Check your images folder.")

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "models/faiss.index")
np.save("models/image_paths.npy", np.array(image_paths))

print(f"Index created successfully — {len(image_paths)} images indexed.")