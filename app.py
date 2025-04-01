import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load Model and Tokenizer
model_name = "tinybert"  # Change this if needed
tokenizer = AutoTokenizer.from_pretrained("tinybert_hate_speech")
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define function for prediction
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["Normal", "Hate Speech", "Offensive"]
    return labels[torch.argmax(probabilities)], probabilities.tolist()

# Streamlit UI
st.title("🛡️ Hate Speech Detection Model")
st.write("Enter a text below to classify it.")

# Text Input
user_input = st.text_area("Enter your text here...", "")

if st.button("Detect"):
    if user_input.strip():
        label, probs = predict(user_input)
        st.write(f"**Prediction:** {label}")
        st.write(f"**Probabilities:** {probs}")
    else:
        st.warning("⚠️ Please enter some text.")

# Footer
st.markdown("---")
st.markdown("🚀 Built with Streamlit & Transformers")

