# Spam Email Classifier

A machine learning web application that classifies email and SMS messages as spam or legitimate (ham) using a Support Vector Machine (SVM) model. The application is built with Streamlit and deployed on Streamlit Cloud.

Live Demo: https://spam-classifiergit-a7btrw294fazn9xn5uora2.streamlit.app

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Machine Learning Approach](#machine-learning-approach)
- [Model Comparison](#model-comparison)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [How It Works](#how-it-works)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## Overview

Spam detection is one of the most classic and practical applications of machine learning in natural language processing (NLP). This project builds an end-to-end spam classifier that takes any text message as input and predicts whether it is spam or a legitimate message.

The model is trained on a labeled dataset of SMS messages and uses TF-IDF vectorization combined with a LinearSVC (Support Vector Machine) classifier to make predictions. The entire pipeline from raw text to prediction is wrapped in an interactive web application built using Streamlit.

---

## Dataset

- Name: UCI SMS Spam Collection Dataset
- Source: Kaggle (UCI Machine Learning Repository)
- Size: 5,572 messages
- Classes:
  - Ham (legitimate): 4,825 messages
  - Spam: 747 messages
- Format: CSV with two columns, Category (label) and Message (text)

---

## Tech Stack

- Python 3
- Scikit-learn — machine learning models and TF-IDF vectorization
- NLTK — natural language processing and stopword removal
- Pandas — data loading and preprocessing
- Joblib — saving and loading trained model files
- Streamlit — frontend web application
- Streamlit Cloud — free deployment platform

---

## Machine Learning Approach

The following steps are followed in the machine learning pipeline:

1. Data Loading
   The dataset is loaded using Pandas and the columns are renamed for consistency.

2. Text Preprocessing
   Each message goes through the following cleaning steps:
   - Convert to lowercase
   - Remove punctuation and special characters using regular expressions
   - Remove common English stopwords using NLTK

3. Feature Extraction
   The cleaned text is converted into numerical features using TF-IDF (Term Frequency Inverse Document Frequency) Vectorization with a vocabulary size of 5,000 features. TF-IDF captures how important a word is in a message relative to the entire dataset.

4. Model Training
   The data is split into 80% training and 20% testing. Two models were trained and compared:
   - Multinomial Naive Bayes
   - LinearSVC (Support Vector Machine)

5. Model Saving
   The best performing model and the TF-IDF vectorizer are saved as .pkl files using Joblib so they can be loaded directly by the Streamlit app without retraining.

---

## Model Comparison

Both models were evaluated on the same test set and compared on accuracy, precision, recall, and F1-score.

| Model                  | Accuracy |
|------------------------|----------|
| Multinomial Naive Bayes | ~97.8%  |
| LinearSVC (SVM)         | ~98.5%  |

LinearSVC was selected as the final model because it achieved higher accuracy and performed better at correctly identifying spam messages with fewer false positives, meaning legitimate messages are less likely to be incorrectly flagged as spam.

---

## Project Structure
```
spam-classifier/
│
├── app.py                  # Streamlit web application
├── train_model.py          # Script for training and saving the model
├── model.pkl               # Trained LinearSVC model (generated after training)
├── vectorizer.pkl          # Fitted TF-IDF vectorizer (generated after training)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## How to Run Locally

Follow these steps to run the project on your own machine.

1. Clone the repository
```
   git clone https://github.com/iisha04/spam-classifier.git
   cd spam-classifier
```

2. Install dependencies
```
   pip install -r requirements.txt
```

3. Train the model
   Place spam.csv in the project folder, then run:
```
   python train_model.py
```
   This will generate model.pkl and vectorizer.pkl in the same folder.

4. Run the app
```
   python -m streamlit run app.py
```

5. Open your browser and go to http://localhost:8501

---

## How It Works

When a user enters a message in the application:

1. The text is preprocessed (lowercased, cleaned, stopwords removed)
2. The TF-IDF vectorizer transforms the cleaned text into a numerical feature vector
3. The trained LinearSVC model predicts whether the message is spam or ham
4. The application displays the result along with a confidence score derived from the model's decision function

---

## Results

The final LinearSVC model achieves the following performance on the test set:

| Metric    | Ham    | Spam   |
|-----------|--------|--------|
| Precision | 0.99   | 0.96   |
| Recall    | 0.99   | 0.95   |
| F1-Score  | 0.99   | 0.96   |
| Accuracy  | 98.5%  |        |

---

## Future Improvements

- Train on a larger and more diverse email dataset such as the Enron Email Dataset for better generalization
- Experiment with deep learning models such as LSTM or BERT for improved accuracy
- Add support for bulk classification by allowing users to upload a CSV file of messages
- Display a word cloud of the most common spam keywords
- Add multilingual support for non-English messages
