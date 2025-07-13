import torch
from transformers import GPT2Model, GPT2Tokenizer
from yolo.model.yolo import create_model
from omegaconf import OmegaConf
from pathlib import Path


def load_config(path):
    return OmegaConf.load(Path(path))


# Load config (adjust if path differs)
config = load_config("yolo/config/model/rd-9c.yaml")

# Build YOLO model from config without loading weights
model = create_model(config, weight_path=False)

# Load GPT-2 (default is gpt2)
gpt2 = GPT2Model.from_pretrained("gpt2")
embedding = gpt2.wte.weight.data  # shape: (vocab_size, hidden_dim)
print(f"📏 GPT-2 embedding shape: {embedding.shape}")

# Find the RD dictionary layer (e.g., `dict_embed`)
dict_layer = None
for layer in model.model:
    if hasattr(layer, "dict_embed"):
        dict_layer = layer
        break

if dict_layer is None:
    raise RuntimeError("❌ No dictionary layer (dict_embed) found in the model")

# Resize and copy GPT-2 embeddings to dictionary
with torch.no_grad():
    trimmed_embed = embedding[:dict_layer.dict_embed.weight.shape[0]]
    dict_layer.dict_embed.weight.copy_(trimmed_embed)
    print(f"✅ Copied {trimmed_embed.shape[0]} embeddings to dict_embed")

# Save the new model weights
torch.save({"model_state_dict": model.model.state_dict()}, "weights/v9-c-llm.pt")
print("🎉 Saved to weights/v9-c-llm.pt")