# 💬 NLP Sentiment Analysis using XGBoost

## 📌 Project Overview

This project is a **Natural Language Processing (NLP) based Sentiment Analysis System** that classifies user reviews or text into three sentiment categories:

* 😊 Positive
* 😐 Neutral
* 😞 Negative

The model is trained using advanced text preprocessing techniques, TF-IDF feature extraction, SMOTE for handling class imbalance, and an XGBoost classifier. A user-friendly web application is built using Streamlit for real-time sentiment prediction.

---

# 🚀 Features

✅ Text Cleaning and Preprocessing

✅ Tokenization

✅ Stopword Removal

✅ Lemmatization

✅ TF-IDF Vectorization

✅ SMOTE for Class Balancing

✅ XGBoost Classification Model

✅ Probability-Based Predictions

✅ Interactive Streamlit Web Application

✅ Confidence Score Display

✅ Real-Time Sentiment Analysis

---

# 🛠️ Technologies Used

| Technology   | Purpose                    |
| ------------ | -------------------------- |
| Python       | Programming Language       |
| NLTK         | Text Preprocessing         |
| Pandas       | Data Handling              |
| NumPy        | Numerical Operations       |
| Scikit-Learn | Machine Learning Utilities |
| TF-IDF       | Feature Extraction         |
| XGBoost      | Classification Model       |
| SMOTE        | Class Imbalance Handling   |
| Joblib       | Model Serialization        |
| Streamlit    | Web Application            |

---

# 📂 Project Structure

```text
NLP--Sentiment-Analysis/

│
├── app.py
├── NLP.ipynb
├── xgboost_sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
├── preprocess_function.pkl
├── requirements.txt
└── README.md
```

---

# 🔄 Project Workflow

### Step 1: Data Collection

* Load review dataset.
* Explore sentiment distribution.

### Step 2: Text Preprocessing

The text undergoes several preprocessing steps:

1. Convert text to lowercase
2. Remove URLs
3. Remove special characters
4. Tokenization
5. Stopword removal
6. Lemmatization


---

### Step 3: Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical vectors suitable for machine learning models.

---

### Step 4: Handle Class Imbalance

SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the sentiment classes before training.

Benefits:

* Reduces bias toward majority classes
* Improves classification performance

---

### Step 5: Model Training

The XGBoost classifier is trained using TF-IDF features.

Advantages of XGBoost:

* High accuracy
* Fast training
* Handles large datasets efficiently
* Robust against overfitting

---

### Step 6: Model Serialization

Trained artifacts are saved using Joblib:

```python
joblib.dump(model, 'xgboost_sentiment_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
```

---

### Step 7: Streamlit Deployment

The trained model is integrated into a Streamlit web application that allows users to:

* Enter custom reviews
* Analyze sentiment instantly
* View confidence scores
* Inspect probability distributions
---

---

# 🎯 Sentiment Classes

| Class    | Meaning                          |
| -------- | -------------------------------- |
| Positive | Positive opinion or satisfaction |
| Neutral  | Neither positive nor negative    |
| Negative | Dissatisfaction or criticism     |

---

# 📈 Model Pipeline

```text
Input Text
     │
     ▼
Text Preprocessing
     │
     ▼
TF-IDF Vectorization
     │
     ▼
XGBoost Model
     │
     ▼
Sentiment Prediction
     │
     ▼
Probability Scores
     │
     ▼
Streamlit Interface
```

# 👨‍💻 Author
** Mr. Zero **
