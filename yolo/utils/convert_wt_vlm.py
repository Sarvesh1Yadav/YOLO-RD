import torch
from transformers import CLIPVisionModel, CLIPImageProcessor
import sys
import os
sys.path.append(os.path.abspath("."))  # Adds project root to import path

from yolo.model.yolo import create_model  # ✅ correct import
from omegaconf import OmegaConf
from pathlib import Path

def load_config(path):
    return OmegaConf.load(Path(path))

# Load config (adjust path if needed)
config = load_config("yolo/config/model/rd-9c.yaml")

# Build the YOLO model from YAML config
model = create_model(config, weight_path=False)

# Load CLIP-ViT vision encoder
clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
clip_weights = clip_model.state_dict()

# Get YOLO model weights
model_weights = model.state_dict()

# Replace overlapping weights
count = 0
for name in clip_weights:
    if name in model_weights and clip_weights[name].shape == model_weights[name].shape:
        model_weights[name] = clip_weights[name]
        count += 1

print(f"✅ Replaced {count} matching weights from CLIP")

# Save VLM-initialized weights
torch.save({"model_state_dict": model_weights}, "weights/v9-c-vlm.pt")
print("🎉 Saved to weights/v9-c-vlm.pt")
