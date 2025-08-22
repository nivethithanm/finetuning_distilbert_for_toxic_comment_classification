# DistilBERT Fine-Tuning for Toxic Comment Classification

A fine-tuned DistilBERT model for multi-label toxicity classification, trained to detect hate speech and offensive content in online comments.

## Project Overview

This project fine-tunes DistilBERT-base-uncased on the HateXplain dataset for automated content moderation. The model classifies text into two categories: **hatespeech** and **offensive**, using LoRA (Low-Rank Adaptation) for efficient training.

## Model Details

- **Base Model**: distilbert-base-uncased (66M parameters)
- **Task**: Multi-label binary classification
- **Labels**: `hatespeech`, `offensive`
- **Dataset**: HateXplain (~20k social media posts)
- **Training**: LoRA fine-tuning

## Quick Start

### Model Usage
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("nivethithan-m/distilbert-hatexplain")
tokenizer = AutoTokenizer.from_pretrained("nivethithan-m/distilbert-hatexplain")
model.eval()

text = "Your text here"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits
thresholds = [0.5, 0.5]
predictions = (torch.sigmoid(logits) > torch.tensor(thresholds)).numpy()
labels = ["hatespeech", "offensive"]
print(dict(zip(labels, predictions[0])))
```

## Streamlit Demo

This project provides a Streamlit application that utilizes a fine-tuned DistilBERT model for multi-label toxicity classification. The model is capable of predicting whether a given text is "hatespeech", "offensive", or "normal, along with confidence scores for each label.

```bash
cd demo
pip install -r requirements.txt
streamlit run src/app.py
```

## Project Structure

```
demo
├── src
│   ├── app.py          # Main entry point for the Streamlit application
│   └── utils.py        # Utility functions for model loading and inference
├── requirements.txt     # List of dependencies
└── README.md            # Project documentation
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd streamlit-distilbert-demo
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run src/app.py
```

This will start the Streamlit server and open the application in your default web browser.

## Usage

1. Enter the text you want to classify in the input box.
2. Click the "Classify" button to run inference.
3. The application will display the predicted labels along with their confidence scores.

## Model Information

The application uses a fine-tuned DistilBERT model hosted on Hugging Face. The model has been trained to classify text into three categories: "hatespeech", "offensive", or "normal.

## Acknowledgements

- Hugging Face Transformers for the pre-trained models.
- Streamlit for the interactive web application framework.


## 📜 License

This project is open source. Please check the dataset license for HateXplain usage terms.
