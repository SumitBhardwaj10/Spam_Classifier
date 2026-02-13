import streamlit as st
import joblib
import regex as re
import emoji
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Spam Shield AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. SETUP & ASSETS ---
# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stem = PorterStemmer()
stop_words = set(stopwords.words("english"))

# --- 3. CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 10px;
        border: 1px solid #d1d5db;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1e3a8a;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. FUNCTIONS ---

@st.cache_resource
def load_models():
    """Load models with caching to improve performance"""
    try:
        vec = joblib.load("Model/Spam_vectorizer.pkl")
        mod = joblib.load("Model/Spam_classifier.pkl")
        return vec, mod
    except FileNotFoundError:
        st.error("🚨 Model files not found! Please ensure 'Model/Spam_vectorizer.pkl' and 'Model/Spam_classifier.pkl' exist.")
        return None, None

def data_cleaning(text):
    text = emoji.demojize(text) 
    
    text = text.lower() # lowercase
    text = re.sub(r'<[^>]+>', ' ', text)  # remove headers
    text = re.sub(r"http\S+|www\S+", "URL", text)   # change website to URL
    text = re.sub(r"\d+", "NUMBERS", text)   # remove numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove punctuation/special chars
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra space

    filtered_text = []
    for word in text.split(" "):
        if word in stop_words:
            continue
        filtered_text.append(stem.stem(word))
    
    return " ".join(filtered_text)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2058/2058768.png", width=100)
    st.title("Spam Shield")
    st.markdown("---")
    st.write("### How to use:")
    st.write("1. Copy the content of an email.")
    st.write("2. Paste it into the text box.")
    st.write("3. Click **Analyze Email**.")
    st.markdown("---")
    st.caption("Powered by Scikit-Learn & NLTK")

st.markdown("<h1 style='text-align:center;'>📧 Email Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:grey;'>Paste your email below to detect if it's safe or spam.</p>", unsafe_allow_html=True)

vectorizer, model = load_models()

email_input = st.text_area("Content", height=250, placeholder="Type or paste your email content here...", label_visibility="collapsed")

col1, col2, col3,col4,col5 = st.columns([1, 1,2,0.5,1])

with col3:
    analyze_button = st.button("🔍 Analyze Email")

if analyze_button:
    if not vectorizer or not model:
        st.error("Models failed to load.")
    elif not email_input:
        st.warning("⚠️ Please paste some text to analyze.")
    else:
        with st.spinner("Analyzing text patterns..."):
            time.sleep(0.8)
            clean_email = data_cleaning(email_input)
            text_vectorized = vectorizer.transform([clean_email])
            prediction = model.predict(text_vectorized)
            confidence=model.predict_proba(text_vectorized)[:,1]
            confidence=round(confidence[0]*100,2)
            st.markdown("---")
            
            if prediction == 0:
                st.balloons()
                st.success("## ✅ Safe Email")
                st.write("This email appears to be **Ham** (Legitimate).")
                st.metric("Confindence",confidence)
            else:
                st.error("## 🚨 Spam Detected")
                st.write("This email shows strong patterns of being **Spam**.")
                st.metric("Confindence",confidence)
    
                