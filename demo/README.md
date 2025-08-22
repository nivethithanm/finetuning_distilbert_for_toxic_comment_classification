# Streamlit DistilBERT Toxicity Classification Demo

This project provides a Streamlit application that utilizes a fine-tuned DistilBERT model for multi-label toxicity classification. The model is capable of predicting whether a given text is "hatespeech", "offensive", or "normal, along with confidence scores for each label.

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