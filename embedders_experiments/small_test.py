from transformers import AutoTokenizer, AutoModel
import torch

model_path = "/lnet/work/people/sajdokova/models/bge-multilingual-gemma2"

model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)

texts = ["Ahoj světe!", "Test embedding on GPU"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

with torch.no_grad():
    out = model(**inputs)

print("✅ Embedding shape:", out.last_hidden_state.shape)
