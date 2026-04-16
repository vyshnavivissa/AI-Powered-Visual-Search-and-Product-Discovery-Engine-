import torch
import os

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables (lazy loading)
model = None
processor = None


def load_model():
    global model, processor

    if model is None:
        print("🔄 Loading CLIP model...")

        from transformers import CLIPProcessor, CLIPModel

        model_name = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
        token = os.getenv("HF_TOKEN")

        # Load model
        model = CLIPModel.from_pretrained(
            model_name,
            token=token
        ).to(device)

        # Load processor
        processor = CLIPProcessor.from_pretrained(
            model_name,
            token=token
        )

        print(f"✅ Model loaded successfully on {device}")

    return model, processor


def get_image_embedding(image):
    model, processor = load_model()

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

    # Normalize (important for similarity search)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy().tolist()


def get_text_embedding(text):
    model, processor = load_model()

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    # Normalize
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy().tolist()