from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import io
from typing import Optional
from backend.model import get_image_embedding, get_text_embedding
from backend.search import search_similar
from backend.llm import extract_filters
from backend.search import apply_filters
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve images statically so frontend can display thumbnails
app.mount("/images", StaticFiles(directory=r"C:\Users\xx\OneDrive\Desktop\visual_search\data\fashion"), name="images")

@app.get("/")
def home():
    return {"message": "Backend is running successfully"}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        embedding = get_image_embedding(image)
        results = search_similar(embedding)
        return {
            "filename": file.filename,
            "similar_images": results
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/search/")
async def search(
    file: Optional[UploadFile] = File(None),
    query: Optional[str] = Form(None)
):
    try:
        # CASE 1: Image + Text — combine both embeddings
        if file and query:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            image_emb = np.array(get_image_embedding(image))
            text_emb  = np.array(get_text_embedding(query))

            # Weighted average: tune alpha (0.0 = text only, 1.0 = image only)
            alpha = 0.6
            combined_emb = alpha * image_emb + (1 - alpha) * text_emb
            # Normalize so cosine similarity stays valid
            combined_emb = combined_emb / np.linalg.norm(combined_emb)

            results = search_similar(combined_emb.tolist())

            # Optionally also apply metadata filters from LLM
            filters = extract_filters(query)
            print("Extracted filters:", filters)   # <-- add this to debug

            if filters:
                filtered_results = apply_filters(results, filters)
                # Fallback: if filters wiped everything out, return unfiltered
                if not filtered_results:
                    print("Warning: filters returned no results, using unfiltered")
                    filtered_results = results
            else:
                filtered_results = results

            return {
                "type": "image + text",
                "filters": filters,
                "results": filtered_results
            }

        # CASE 2: Only Image
        elif file:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            emb = get_image_embedding(image)
            results = search_similar(emb)
            return {"type": "image", "results": results}

        # CASE 3: Only Text
        elif query:
            emb = get_text_embedding(query)
            results = search_similar(emb)
            return {"type": "text", "raw_results": results}

        else:
            return {"error": "Provide at least an image or text query"}

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}  # detailed error