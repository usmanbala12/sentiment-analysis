import streamlit as st
import joblib
import re
from bs4 import BeautifulSoup

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('./model/sentiment_model.joblib')
    vectorizer = joblib.load('./model/tfidf_vectorizer.joblib')
    return model, vectorizer

# Preprocessing function (match your notebook's preprocessing)
def preprocess_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

st.title("Twitter Sentiment Analyzer")
st.write("Analyze the sentiment of tweets using Machine Learning")

model, vectorizer = load_model()

# Text input
user_input = st.text_area("Enter a tweet to analyze:", height=100)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Preprocess and predict
        processed = preprocess_text(user_input)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]
        
        if prediction == 1:
            st.success("**Positive Sentiment**")
        else:
            st.error("**Negative Sentiment**")
        
        st.write(f"Confidence: {max(probability) * 100:.1f}%")
    else:
        st.warning("Please enter some text to analyze")