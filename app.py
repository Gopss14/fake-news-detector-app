import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ---------- Custom Background and Styling ----------
# Add this at the top of your app.py after `st.set_page_config(...)`
st.markdown("""
    <style>
    html, body, .stApp {
        height: 100%;
        margin: 0;
        padding: 0;
        overflow: hidden;
    }

    .stApp {
        background: linear-gradient(to right, #141E30, #243B55);
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 800px;
    }

    .stTextArea textarea {
        background-color: #1e1e2f;
        color: white;
        border-radius: 8px;
    }

    .stButton > button {
        background-color: #4F8BF9;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
    }

    .stButton > button:hover {
        background-color: #1f5edb;
        color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)



# ---------- App Title & Description ----------
st.markdown("<h1 style='text-align: center;'>📰 Fake News Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Enter a news article to check whether it's <b>Real</b> or <b>Fake</b>.</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------- Input & Prediction ----------
news_text = st.text_area("📝 Enter News Content", height=200, placeholder="Type or paste the news article here...")

if st.button("🔍 Predict", use_container_width=True):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news content to analyze.")
    else:
        transformed = vectorizer.transform([news_text])
        result = model.predict(transformed)[0]

        st.markdown("---")
        if result == 1:
            st.success("✅ This news article appears to be **REAL**.")
        else:
            st.error("❌ This news article appears to be **FAKE**.")
