# Resume Classification System

This project is an NLP-based Resume Classification system built using Python and Scikit-learn.

## Features
- Upload resume files (PDF/DOCX)
- Extract text from resume
- Clean and preprocess text
- Convert text to TF-IDF features
- Predict job category using Linear SVM
- Display prediction confidence

## Technologies Used
- Python
- Scikit-learn
- TF-IDF
- Streamlit
- NLP

## Model
The model was trained using multiple classifiers including:
- Logistic Regression
- Naive Bayes
- Linear SVM

LinearSVC achieved **95% cross-validation accuracy** and was selected as the final model.
