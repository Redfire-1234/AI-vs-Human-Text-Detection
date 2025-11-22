import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("AI vs Human Text Classifier")

text = st.text_area("Enter text to classify")

HF_MODEL_REPO = "Redfire-1234/bert-ai-human-model"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO, use_auth_token=False)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_REPO, use_auth_token=False)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

if st.button("Predict"):
    if not text.strip():
        st.error("Please enter some text before prediction.")
        st.stop()

    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        predicted_class = int(torch.argmax(outputs.logits, dim=1))

    label_map = {0: "Human", 1: "AI"}
    st.subheader(f"Prediction: {label_map[predicted_class]}")
    st.progress(int(probs[predicted_class] * 100))
    st.write(f"Confidence: {probs[predicted_class]:.3f}")



