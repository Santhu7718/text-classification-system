# Text Classification System

## Project Overview
This project is a Text Classification System developed as part of the AI/ML Engineer Intern technical assignment for Ardentix.  
The goal is to classify SMS messages into **Spam** or **Ham** using machine learning techniques.

---

## Dataset
- **Dataset Name:** SMS Spam Collection
- **Source:** UCI Machine Learning Repository
- **Classes:** Spam, Ham
- **Total Samples:** 5572

---

## Approach & Pipeline

### 1. Data Loading
The dataset was loaded using Pandas and structured into two columns:
- `label` – target class (spam/ham)
- `text` – raw SMS message

---

### 2. Text Preprocessing
The following preprocessing steps were applied:
- Converted text to lowercase
- Removed numbers and punctuation
- Tokenized text using NLTK
- Removed English stopwords

A new column `clean_text` was created after preprocessing.

---

### 3. Feature Extraction
Text data was converted into numerical format using **TF-IDF Vectorization** with a maximum of 3000 features.

**Why TF-IDF?**
- Reduces impact of common words
- Highlights informative terms
- Works well with linear models

---

### 4. Model Training
Two machine learning models were trained:
- **Multinomial Naive Bayes**
- **Logistic Regression**

A stratified 80–20 train-test split was used to handle class imbalance.

---

### 5. Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

**Logistic Regression Performance:**
- Accuracy: ~96%
- High precision for spam detection
- Minimal false positives

---

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Matplotlib, Seaborn

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook

---

## Model Selection Rationale

Two models were selected for this project:

### Naive Bayes
Naive Bayes is well-suited for text classification problems because it performs efficiently on high-dimensional sparse data such as TF-IDF vectors. It assumes feature independence, which works surprisingly well for text-based problems.

### Logistic Regression
Logistic Regression was chosen as it is a strong linear classifier that provides better generalization and probabilistic interpretation. It performs well with TF-IDF features and helps reduce false positives in spam detection.

After evaluation, Logistic Regression was preferred due to its higher precision and balanced performance.

---

## Observations & Insights

- Logistic Regression achieved approximately **96% accuracy** on the test set.
- The model demonstrated **high precision for spam detection**, meaning legitimate messages were rarely misclassified as spam.
- Naive Bayes trained faster but showed slightly lower generalization compared to Logistic Regression.
- The confusion matrix showed zero false positives for ham messages, which is critical for real-world spam filtering systems.

Overall, Logistic Regression was selected as the final model due to its reliability and robustness.
