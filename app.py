import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources (only runs once)
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True) 
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    return True

download_nltk_resources()

# ===========================
# PREPROCESSING FUNCTION (SAME AS TRAINING)
# ===========================

# Initialize lemmatizer and stopwords ONCE
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Clean and preprocess text - EXACT SAME FUNCTION AS TRAINING.
    CRITICAL: Any changes here must match the training notebook.
    """
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    
    # Remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z]", " ", text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

# ===========================
# LOAD MODEL (CACHED)
# ===========================

@st.cache_resource
def load_models():

    try:
        model = joblib.load('xgboost_sentiment_model.pkl')

        tfidf = joblib.load('tfidf_vectorizer.pkl')

        label_encoder = joblib.load('label_encoder.pkl')

        return model, tfidf, label_encoder

    except FileNotFoundError:
        st.error("❌ Model files not found!")
        st.stop()

    except Exception as e:
        st.error(f"❌ Error loading files: {e}")
        st.stop()


model, tfidf, label_encoder = load_models()

# ===========================
# PAGE CONFIG
# ===========================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

# ===========================
# HEADER
# ===========================

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
    💬 Sentiment Analysis App
    </h1>
    <p style='text-align: center; font-size: 1.2em; color: #555;'>
    Analyze the sentiment of your text instantly
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# MAIN LAYOUT
# ===========================

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Enter Your Text")
    
    # Sample examples
    sample = st.selectbox(
        "Try an example:",
        [
            "",
            "This product is amazing! Best purchase ever!",
            "Worst experience ever, total waste of money",
            "It's okay, nothing special",
            "Battery backup is excellent. Great value for money!",
            "Camera quality is very poor, highly disappointed"
        ]
    )
    
    # Text input
    user_input = st.text_area(
        "Enter your review or text here:",
        value=sample,
        height=200,
        placeholder="Type or paste your text here..."
    )
    
    # Buttons
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        predict_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    with btn_col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        st.rerun()

with col2:
    st.markdown("### 📊 Results")
    
    if predict_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing sentiment... ⏳"):
                # Preprocess
                cleaned_text = preprocess_text(user_input)
            
                # TF-IDF transform
                vectorized = tfidf.transform([cleaned_text])
            
                # Sparse → Dense
                vectorized_dense = vectorized.toarray()
            
                # Prediction
                prediction_encoded = model.predict(vectorized_dense)[0]
            
                # Decode label
                prediction = label_encoder.inverse_transform(
                    [prediction_encoded]
                )[0]
            
                # Probabilities
                probabilities = model.predict_proba(
                    vectorized_dense
                )[0]
            
                # Confidence
                confidence = max(probabilities) * 100
            
                # Labels
                class_labels = label_encoder.classes_
            
                # Probability dictionary
                prob_dict = dict(zip(class_labels, probabilities))            
            # Display result
            st.markdown("#### 🎯 Prediction")
            
            if prediction == "Positive":
                st.success(f"😊 **{prediction}** Sentiment")
                st.balloons()
            elif prediction == "Negative":
                st.error(f"😞 **{prediction}** Sentiment")
            else:
                st.info(f"😐 **{prediction}** Sentiment")
            
            st.markdown(f"**Confidence:** {confidence:.1f}%")
            
            # Probability breakdown
            st.markdown("---")
            st.markdown("#### 📈 Probability Breakdown")
            
            # Display probabilities as progress bars
            for label in ['Negative', 'Neutral', 'Positive']:
               prob = float(prob_dict[label])
               color = {
                    'Negative': '🔴',
                    'Neutral': '⚪',
                    'Positive': '🟢'
                }[label]
            
                st.markdown(f"{color} **{label}**")
            
                st.progress(prob)
            
                st.caption(f"{prob*100:.1f}%")
            
            # Debug info (collapsible)
            with st.expander("🔍 Debug Information"):
                st.markdown("**Original Text:**")
                st.code(user_input)
                
                st.markdown("**Preprocessed Text:**")
                st.code(cleaned_text)
                
                st.markdown("**All Probabilities:**")
                for label, prob in prob_dict.items():
                    st.write(f"{label}: {prob:.4f}")
    else:
        st.info("👈 Enter text and click 'Analyze Sentiment' to see results")

# ===========================
# FOOTER
# ===========================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
    <p>Built with Streamlit • Powered by Scikit-Learn</p>
    <p>Model: XGBoost with TF-IDF + SMOTE</p>
    </div>
""", unsafe_allow_html=True)
