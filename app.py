import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

st.title("AI vs Human Text Classifier")

text = st.text_area("Enter text to classify:")

if st.button("Predict"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-ai-human-model")
    model = AutoModelForSequenceClassification.from_pretrained("bert-ai-human-model")
    model.eval()

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        predicted_class = int(torch.argmax(outputs.logits, dim=1))

    label_map = {0: "Human", 1: "AI"}
    st.write(f"**Prediction:** {label_map[predicted_class]}")
    st.write(f"**Confidence:** {probs[predicted_class]:.3f}")

    # LIME explanation
    class_names = ["Human", "AI"]
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_fn(texts):
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.softmax(logits, dim=1).numpy()

    st.write("**Words influencing the prediction:**")
    exp = explainer.explain_instance(text, predict_fn, num_features=10)
    st.write(exp.as_list())
