from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

app = FastAPI()

# تحميل الموديل والتوكنايزر
model_path = "./needs_model"


model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# تحميل ماب الـ labels
with open(f"{model_path}/id2label.json", "r", encoding="utf-8") as f:
    id2label = json.load(f)

# نموذج البيانات اللي جايه من الباك إند
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_id = torch.argmax(probs).item()

    label_info = id2label[str(predicted_id)]
    return {
        "category": label_info["category"],
        "sub_category": label_info["sub_category"],
        "confidence": float(probs[0][predicted_id])
    }

