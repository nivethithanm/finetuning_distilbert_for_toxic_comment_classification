import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model_and_tokenizer():
    model_name = "nivethithan-m/distilbert-hatexplain"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def preprocess_text(text):
    return text.strip()

def run_inference(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.sigmoid(logits).numpy()[0]
    return probabilities

def interpret_predictions(probabilities, threshold=0.5):
    labels_list = ["hatespeech", "offensive"]
    predictions = (probabilities > threshold).astype(int)
    result = dict(zip(labels_list, predictions))
    return {
        'category': ', '.join([label for label, pred in result.items() if pred == 1]),
        'predictions': predictions,
        'probabilities': probabilities,
        'labels': labels_list
    }