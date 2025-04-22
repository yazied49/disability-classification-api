import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

# تحميل الموديل والتوكنايزر
model_path = "C:/Users/Yazied/finall projectt/needs_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# تحميل ماب الـ labels
with open(f"{model_path}/id2label.json", "r", encoding="utf-8") as f:
    id2label = json.load(f)

# نص تجريبي (ممكن تغيره)
text = "This post is discussing challenges and solutions related to ocd under the broader topic of mental health disabilities."

# تجهيز الإدخال
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# التنبؤ
with torch.no_grad():
    outputs = model(**inputs)
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
predicted_id = torch.argmax(probs).item()

# جلب النتيجة
label_info = id2label[str(predicted_id)]
print("\n🧠 Prediction Result:")
print("Category:", label_info["category"])
print("Sub-category:", label_info["sub_category"])
print("Confidence:", float(probs[0][predicted_id]))
