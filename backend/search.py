import faiss
import numpy as np
import pandas as pd

# Load CSV
df = pd.read_csv(r"C:\Users\xx\OneDrive\Desktop\visual_search\data\fashion_subset.csv")
df.columns = df.columns.str.strip()
df["brand"] = df["brand"].fillna("")  # ✅ fill 439 nulls upfront

# Create lookup dictionary
metadata_dict = {}

for _, row in df.iterrows():
    key = str(row["id"]) + ".jpg"

    raw_brand = str(row["brand"]).strip()

    # If brand is empty/nan, derive from first word of product name
    if raw_brand.lower() in ("", "nan", "none"):
        product_name = str(row.get("productDisplayName", ""))
        derived_brand = product_name.split()[0].lower() if product_name else "unknown"
    else:
        derived_brand = raw_brand.lower()

    metadata_dict[key] = {
        "price":        row.get("price", 0),
        "color":        str(row.get("baseColour",  "unknown")).lower(),
        "category":     str(row.get("articleType", "unknown")).lower(),
        "brand":        derived_brand,   # ✅ real brand or derived from product name
        "product_name": str(row.get("productDisplayName", "")),
    }

# ✅ Verify it's working — should print real brand names
print("Sample metadata:")
for k, v in list(metadata_dict.items())[:5]:
    print(f"  {k} → brand: {v['brand']}, color: {v['color']}, category: {v['category']}")

# Load FAISS index
index = faiss.read_index("models/faiss.index")

# Load image paths
image_paths = np.load("models/image_paths.npy")


def search_similar(embedding, top_k=5):
    embedding = np.array(embedding).astype("float32")
    distances, indices = index.search(embedding, top_k)
    return [image_paths[idx] for idx in indices[0]]


def apply_filters(results, filters):
    print("Filters received:", filters)
    filtered = []

    for img in results:
        img = str(img)  # ✅ ensure it's a plain string (numpy strings can cause .get() misses)
        item = metadata_dict.get(img)

        if not item:
            print(f"Warning: No metadata found for '{img}'")
            continue

        # PRICE
        if filters.get("max_price") and item["price"] > filters["max_price"]:
            continue

        # COLOR
        if filters.get("color"):
            if filters["color"].lower() not in item["color"]:
                continue

        # CATEGORY
        if filters.get("category"):
            if filters["category"].lower() not in item["category"]:
                continue

        # BRAND
        if filters.get("brand"):
            brand_query = filters["brand"].lower()
            # Check brand field AND full product name for better matching
            if brand_query not in item["brand"] and brand_query not in item["product_name"].lower():
                continue

        filtered.append({
            "image":        img,
            "image_url":    f"/images/{img}",
            "price":        item["price"],
            "color":        item["color"],
            "category":     item["category"],
            "brand":        item["brand"],
            "product_name": item["product_name"],  # ✅ include for frontend display
        })

    return filtered