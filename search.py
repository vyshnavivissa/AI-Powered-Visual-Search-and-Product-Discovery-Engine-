import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os 
# load .env
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")

df = pd.read_csv("fashion_subset.csv")
df.columns = df.columns.str.strip()

print("CSV Columns:", df.columns.tolist())  # debug

metadata_dict = {}

for _, row in df.iterrows():
    key = str(row["id"]) + ".jpg"

    product_name = str(row.get("productDisplayName", ""))

    # No brand column — derive from first word of product name
    derived_brand = product_name.split()[0].lower() if product_name else "unknown"

    metadata_dict[key] = {
        "price":        row.get("price", 0),
        "color":        str(row.get("baseColour", "unknown")).lower(),
        "category":     str(row.get("articleType", "unknown")).lower(),
        "brand":        derived_brand,
        "product_name": product_name,
    }

print("Sample metadata:")
for k, v in list(metadata_dict.items())[:5]:
    print(f"  {k} → brand: {v['brand']}, color: {v['color']}, category: {v['category']}")

index = faiss.read_index("faiss.index")
image_paths = np.load("image_paths.npy")

def search_similar(embedding, top_k=20):
    embedding = np.array(embedding).astype("float32").reshape(1, -1)

    distances, indices = index.search(embedding, top_k)

    return [image_paths[idx] for idx in indices[0]]
def apply_filters(results, filters):
    print("Filters received:", filters)
    filtered = []

    for img in results:
        img = str(img)
        item = metadata_dict.get(img)

        if not item:
            print(f"Warning: No metadata found for '{img}'")
            continue

        if filters.get("max_price") and item["price"] > filters["max_price"]:
            continue

        if filters.get("color"):
            if filters["color"].lower() not in item["color"]:
                continue

        if filters.get("category"):
            if filters["category"].lower() not in item["category"]:
                continue

        if filters.get("brand"):
            brand_query = filters["brand"].lower()
            if brand_query not in item["brand"] and brand_query not in item["product_name"].lower():
                continue

        filtered.append({
            "image":        img,
            "image_url":    f"{BACKEND_URL}/images/{img}",
            "price":        item["price"],
            "color":        item["color"],
            "category":     item["category"],
            "brand":        item["brand"],
            "product_name": item["product_name"],
        })

    return filtered