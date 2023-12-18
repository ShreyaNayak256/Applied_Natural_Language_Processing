# CSCI 544: Assignment 3 Report

## Grade: 100/100

## Overview
This assignment focuses on advanced text processing and classification using various machine learning models, including Support Vector Machines (SVM), Perceptron, Multi-Layer Perceptron (MLP), and Recurrent Neural Networks (RNNs).

## Task 1: Reading and Cleaning Data
- Utilized pandas to import the dataset.
- Conducted text-based data cleaning, retaining only the review label and text review for analysis.
- Used Google Colab for its GPU capabilities&#8203;``【oaicite:4】``&#8203;.

## Task 2: Data Processing
- TFIDF Transformation: Converted text data into numerical representations using the TFIDF vectorizer.
- Word2Vec Processing: Employed Google's Word2Vec model via Gensim for generating average word vectors&#8203;``【oaicite:3】``&#8203;.

## Task 3: SVM and Perceptron Models
- Trained SVM and Perceptron models using both TF-IDF and Word2Vec data.
- Observations:
  - Perceptron: Higher accuracy with Word2Vec (74%) compared to TF-IDF (73%).
  - SVM: Best performance with TF-IDF, achieving 87% accuracy.
  - Generally, SVM outperformed Perceptron across feature types&#8203;``【oaicite:2】``&#8203;.

## Task 4: FeedForward Neural Network (MLP)
- Processed data for MLP training using PyTorch.
- Achieved around 82% accuracy with averaged Word2Vec features and 75% accuracy with features of the first 10 words of reviews&#8203;``【oaicite:1】``&#8203;.

## Task 5: Recurrent Neural Network (RNN)
- Different RNN architectures were explored: Simple RNN, LSTM, and GRU.
- Preprocessed data using a custom `vectorize_reviews` function.
- Accuracies:
  - Simple RNN: 75.2%
  - LSTM: 78.1%
  - GRU: 78.4%, the best-performing model among the three.
- Noted that models were trained beyond overfitting as per instructor's advice&#8203;``【oaicite:0】``&#8203;.

---

*Note: This report is a concise summary of the assignment. For detailed code and analysis, refer to the Jupyter notebooks attached.*
