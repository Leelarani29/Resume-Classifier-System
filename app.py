import streamlit as st
import joblib
import re
import numpy as np
from PyPDF2 import PdfReader
from docx import Document

# Page configuration
st.set_page_config(page_title="Resume Classifier", layout="centered")

# Load trained model and vectorizer
model = joblib.load("resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text_from_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text


def extract_text_from_docx(file):
    text = ""
    doc = Document(file)
    for para in doc.paragraphs:
        text += para.text + " "
    return text


st.title("📄 Resume Classification System")
st.write("Upload a resume file (PDF or DOCX) to predict its job category.")

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)

# Show supported categories
st.write("**Supported Job Categories:**")
st.write(", ".join(model.classes_))


if uploaded_file is not None:

    file_type = uploaded_file.name.split(".")[-1].lower()

    try:
        if file_type == "pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = extract_text_from_docx(uploaded_file)

    except:
        st.error("❌ Unable to read the file. Please upload a valid resume.")
        st.stop()

    cleaned_text = clean_text(resume_text)

    # Resume validation
    word_count = len(cleaned_text.split())

    st.write(f"Detected resume length: **{word_count} words**")

    if word_count < 100:
        st.error("❌ This file does not appear to be a valid resume.")
    else:

       
        vectorized = vectorizer.transform([cleaned_text])

       
        decision_scores = model.decision_function(vectorized)

       
        exp_scores = np.exp(decision_scores)
        probabilities = exp_scores / np.sum(exp_scores)

        confidence = np.max(probabilities) * 100
        prediction = model.classes_[np.argmax(probabilities)]

        # Display results
        st.success(f"Predicted Category: **{prediction}**")
        st.info(f"Model Confidence: **{confidence:.2f}%**")

        # Optional: show extracted resume text
        with st.expander("View Extracted Resume Text"):
            st.write(resume_text[:2000])