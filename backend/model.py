import torch
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print(f'Model loaded successfully on {device}')

def get_image_embedding(image):
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
    if hasattr(image_features, "pooler_output"):
        image_features = image_features.pooler_output

    return image_features.cpu().numpy().tolist()
def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        # Use text_model directly
        output = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        # pooler_output is the [CLS] token — shape: (1, 512)
        text_features = output.pooler_output
    return text_features.cpu().numpy().tolist()