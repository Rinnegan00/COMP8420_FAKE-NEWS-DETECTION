# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
model_path = "saved_bert_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# FastAPI setup
app = FastAPI(title="Fake News Detector")

# Request body format
class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: TextInput):
    inputs = tokenizer(
        input.text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    label = "true" if prediction == 1 else "false"
    return {"prediction": label}
