import os
import torch
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel

# ✅ Load env variables
load_dotenv()

# ✅ Reduce CPU memory usage (IMPORTANT for Render)
os.environ["OMP_NUM_THREADS"] = "1"

# ✅ Force CPU (Render has no GPU)
device = "cpu"

# ✅ HuggingFace token
HF_TOKEN = os.getenv("HF_TOKEN")

# ✅ Load model (optimized)
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    token=HF_TOKEN,
    torch_dtype=torch.float32
).to(device)

# ✅ Load processor
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    token=HF_TOKEN
)

# ✅ Set eval mode (saves memory)
model.eval()

print(f"✅ Model loaded successfully on {device}")

# ============================
# 🔹 IMAGE EMBEDDING FUNCTION
# ============================
def get_image_embedding(image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        image_features = model.get_image_features(**inputs)

    return image_features.cpu().numpy().tolist()


# ============================
# 🔹 TEXT EMBEDDING FUNCTION
# ============================
def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_features = model.get_text_features(**inputs)

    return text_features.cpu().numpy().tolist()