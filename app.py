import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_classifier (3).pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Email Spam Classifier")
text_input = st.text_area("Enter the email content:")

if st.button("Classify"):
    if text_input:
        text_vec = vectorizer.transform([text_input])
        prediction = model.predict(text_vec)[0]
        label = "Spam" if prediction == 1 else "Ham"
        st.write(f"**Prediction:** {label}")
    else:
        st.warning("Please enter some text.")
