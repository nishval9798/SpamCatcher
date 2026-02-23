import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (first time only)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model & vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text preprocessing
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i.isalnum()]
    tokens = [i for i in tokens if i not in stop_words]
    tokens = [lemma.lemmatize(i) for i in tokens]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="SpamCatcher", page_icon="📧")

st.title("📧 SpamCatcher")
st.subheader("Email / SMS Spam Detection System")
st.write("Enter a message below to check whether it is **Spam** or **Not Spam**.")

input_text = st.text_area("Enter your message here:")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        transformed_text = transform_text(input_text)
        vector_input = vectorizer.transform([transformed_text])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")