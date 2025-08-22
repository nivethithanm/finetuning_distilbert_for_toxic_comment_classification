import streamlit as st
import numpy as np
from utils import load_model_and_tokenizer, preprocess_text, run_inference, interpret_predictions

# Set page config
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load model and tokenizer with caching"""
    return load_model_and_tokenizer()

def main():
    st.title("🛡️ Toxic Comment Classification Demo")
    st.markdown("This demo uses a fine-tuned DistilBERT model to classify text for toxicity, hate speech, and offensive content.")
    
    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer = load_model()
    
    # Input section
    st.header("Enter Text to Analyze")
    text_input = st.text_area(
        "Type or paste your text here:",
        placeholder="Enter the text you want to analyze for toxicity...",
        height=100
    )
    
    # Threshold slider
    threshold = st.slider(
        "Classification Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Adjust the threshold for binary classification"
    )
    
    if st.button("Analyze Text", type="primary"):
        if text_input.strip():
            # Preprocess text
            processed_text = preprocess_text(text_input)
            
            # Run inference
            with st.spinner("Analyzing..."):
                probabilities = run_inference(model, tokenizer, processed_text)
                results = interpret_predictions(probabilities, threshold)
            
            # Display results
            st.header("Analysis Results")
            
            # Main classification result
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if results['category'] == "Normal":
                    st.success(f"✅ **Classification: {results['category']}**")
                else:
                    st.error(f"⚠️ **Classification: {results['category']}**")
            
            with col2:
                max_prob = max(results['probabilities'])
                st.metric("Max Confidence", f"{max_prob:.2%}")
            
            # Detailed probabilities
            st.subheader("Detailed Probabilities")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                toxic_prob = results['probabilities'][0]
                st.metric(
                    "Toxic",
                    f"{toxic_prob:.2%}",
                    delta=f"{'Above' if toxic_prob > threshold else 'Below'} threshold"
                )
                if toxic_prob > threshold:
                    st.error("🔴 Detected")
                else:
                    st.success("🟢 Not detected")
            
            with col2:
                hate_prob = results['probabilities'][1]
                st.metric(
                    "Hate Speech",
                    f"{hate_prob:.2%}",
                    delta=f"{'Above' if hate_prob > threshold else 'Below'} threshold"
                )
                if hate_prob > threshold:
                    st.error("🔴 Detected")
                else:
                    st.success("🟢 Not detected")
            
            with col3:
                offensive_prob = results['probabilities'][2]
                st.metric(
                    "Offensive",
                    f"{offensive_prob:.2%}",
                    delta=f"{'Above' if offensive_prob > threshold else 'Below'} threshold"
                )
                if offensive_prob > threshold:
                    st.error("🔴 Detected")
                else:
                    st.success("🟢 Not detected")
            
            # Probability bar chart
            st.subheader("Probability Distribution")
            chart_data = {
                'Category': results['labels'],
                'Probability': results['probabilities']
            }
            st.bar_chart(chart_data, x='Category', y='Probability')
            
            # Raw data expander
            with st.expander("Show Raw Data"):
                st.json({
                    'input_text': text_input,
                    'processed_text': processed_text,
                    'probabilities': results['probabilities'].tolist(),
                    'binary_predictions': results['predictions'].tolist(),
                    'threshold_used': threshold,
                    'final_classification': results['category']
                })
        
        else:
            st.warning("Please enter some text to analyze.")
    
    # Information section
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This demo uses a fine-tuned DistilBERT model from HuggingFace:
        
        **Model:** `nivethithan-m/distilbert-hatexplain`
        
        **Categories:**
        - **Toxic**: General toxicity
        - **Hate Speech**: Targeted hate speech
        - **Offensive**: Offensive language
        
        **How it works:**
        1. Text is tokenized using DistilBERT tokenizer
        2. Model outputs probabilities for each category
        3. Probabilities above threshold are classified as positive
        4. Multiple categories can be detected simultaneously
        """)
        
        st.header("Example Texts")
        examples = [
            "I love this beautiful sunny day!",
            "You're such an idiot",
            "All people from that country are bad",
            "This is just stupid"
        ]
        
        for example in examples:
            if st.button(f"Try: '{example[:30]}...'", key=example):
                st.session_state.example_text = example
        
        if hasattr(st.session_state, 'example_text'):
            st.rerun()

if __name__ == "__main__":
    main()
