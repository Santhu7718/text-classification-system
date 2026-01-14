# 📩 Text Classification System (SMS Spam Detection)

## 📌 Project Overview

This project is an **end-to-end Machine Learning–based Text Classification System** designed to classify SMS messages as **Spam** or **Ham (Not Spam)**.  
It demonstrates the complete ML workflow — from data preprocessing and feature extraction to model training, evaluation, and deployment using **Streamlit**.

The project was developed as part of an **AI/ML Engineer technical assignment** and follows industry-standard practices.

---

## 🎯 Problem Statement

Spam messages cause inconvenience, fraud, and security risks.  
The goal of this project is to **automatically detect spam SMS messages** using Natural Language Processing (NLP) and Machine Learning techniques.

---

## 📂 Project Structure

```text
text-classification-system/
│
├── data/
│   └── SMSSpamCollection
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── App.py
├── requirements.txt
├── .gitignore
└── README.md

---
```
## 📊 Dataset Information

- **Dataset Name:** SMS Spam Collection  
- **Source:** UCI Machine Learning Repository  
- **Total Messages:** 5,572  

### Class Distribution
- **Ham:** 4,825  
- **Spam:** 747  

The dataset is slightly imbalanced and is handled using **stratified train-test splitting** to preserve class proportions.

---

## 🧠 Workflow Explanation

### 1️⃣ Data Loading
- Dataset loaded using **Pandas**
- Tab-separated file format
- Columns:
  - `label` → spam / ham
  - `text` → SMS content

**Notebook:** `01_data_loading.ipynb`

---

### 2️⃣ Text Preprocessing
Text cleaning performed using **NLTK**:
- Lowercasing text
- Removing numbers
- Removing punctuation and special characters
- Tokenization
- Stopword removal
- Reconstructing cleaned text

**Notebook:** `02_text_preprocessing.ipynb`

---

### 3️⃣ Feature Extraction
- TF-IDF Vectorization applied to text
- Maximum features limited to **3000**
- Efficient handling of sparse text data

**Notebook:** `03_feature_extraction.ipynb`

---

### 4️⃣ Model Training
Models trained:
- Multinomial Naive Bayes
- Logistic Regression

**Training Strategy:**
- 80% Training
- 20% Testing
- Stratified split on labels

**Notebook:** `04_model_training.ipynb`

---

### 5️⃣ Model Evaluation
Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

**Notebook:** `05_model_evaluation.ipynb`

Final model selected: **Logistic Regression**  
Reason: Better generalization and fewer false positives.

---
##How to run this locally : 
1.Clone the repository : 

```git clone https://github.com/Santhu7718/text-classification-system.git
cd text-classification-system```

