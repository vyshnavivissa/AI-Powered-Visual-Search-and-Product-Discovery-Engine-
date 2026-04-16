import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
processor = None

def load_model():
    global model, processor

    if model is None:
        from transformers import CLIPProcessor, CLIPModel

        model_name = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")

        print("Loading model...")
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        print(f"Model loaded successfully on {device}")

    return model, processor


def get_image_embedding(image):
    model, processor = load_model()

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])

    if hasattr(image_features, "pooler_output"):
        image_features = image_features.pooler_output

    return image_features.cpu().numpy().tolist()


def get_text_embedding(text):
    model, processor = load_model()

    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        output = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        text_features = output.pooler_output

    return text_features.cpu().numpy().tolist()