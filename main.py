from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import io
from typing import Optional

from model import get_image_embedding, get_text_embedding
from search import search_similar, apply_filters, metadata_dict
from llm import extract_filters
import os 
app = FastAPI()
from dotenv import load_dotenv

# load .env
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIXED PATH
app.mount("/images", StaticFiles(directory="fashion"), name="images")


@app.get("/")
def home():
    return {"message": "Backend is running successfully"}


@app.post("/search/")
async def search(
    file: Optional[UploadFile] = File(None),
    query: Optional[str] = Form(None)
):
    try:

        # =============================
        # CASE 1: IMAGE + TEXT
        # =============================
        if file and query:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            image_emb = np.array(get_image_embedding(image)).flatten()
            text_emb = np.array(get_text_embedding(query)).flatten()

            alpha = 0.6
            combined_emb = alpha * image_emb + (1 - alpha) * text_emb
            combined_emb = combined_emb / np.linalg.norm(combined_emb)

            results = search_similar(combined_emb.tolist(), top_k=20)

            filters = extract_filters(query) if len(query.split()) > 2 else {}

            filtered_results = apply_filters(results, filters)

            if not filtered_results:
                print("No match → showing similar items")
                filtered_results = apply_filters(results, {})

            return {
                "type": "image + text",
                "filters": filters,
                "results": filtered_results
            }

        # =============================
        # CASE 2: ONLY IMAGE
        # =============================
        elif file:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            emb = get_image_embedding(image)
            results = search_similar(emb, top_k=20)

            full_results = []
            for img in results:
                item = metadata_dict.get(str(img))
                if item:
                    full_results.append({
                        "image": img,
                        "image_url": f"{BACKEND_URL}/images/{img}",
                        "price": item["price"],
                        "color": item["color"],
                        "category": item["category"],
                        "brand": item["brand"],
                        "product_name": item["product_name"]
                    })

            return {
                "type": "image",
                "filters": {},
                "results": full_results
            }

        # =============================
        # CASE 3: ONLY TEXT
        # =============================
        elif query:
            emb = get_text_embedding(query)
            results = search_similar(emb, top_k=20)

            filters = extract_filters(query) if len(query.split()) > 2 else {}

            filtered_results = apply_filters(results, filters)

            if not filtered_results:
                print("⚠️ No match → showing similar items")
                filtered_results = apply_filters(results, {})

            return {
                "type": "text",
                "filters": filters,
                "results": filtered_results
            }

        # =============================
        # NO INPUT
        # =============================
        else:
            return {"error": "Provide image or query"}

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}