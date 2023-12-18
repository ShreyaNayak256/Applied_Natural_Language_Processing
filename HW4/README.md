# CSCI 544: Assignment 4 Report

## Grade: 140/100 (extra credit for Bonus Task)

## Overview
This assignment explores the implementation and comparison of different NLP models including BiLSTM and Transformers for Named Entity Recognition (NER) tasks using various embedding techniques.

## Task 1: BiLSTM Model without Specific Embeddings
- **Data Processing:** Utilized the CoNLL-2003 corpus, creating a `word2idx` vocabulary with a frequency threshold of 2.
- **Model Architecture:** BiLSTMNER model with specific layers and configurations.
- **Training and Evaluation:** Implemented pack and pad sequence functions for better model focus and accuracy.

## Task 2: BiLSTM with Glove-100 Embeddings
- **Embedding Utilization:** Glove embeddings were adapted to account for case sensitivity, enhancing the model's understanding of semantic relationships in data.
- **Model Enhancements:** Additional dimensions for word embeddings were introduced to handle different cases (lowercase, titlecase, uppercase).
- **Performance Insight:** BiLSTM with Glove Embeddings outperformed the model without, due to its superior handling of semantic context and pre-trained representations&#8203;``【oaicite:2】``&#8203;&#8203;``【oaicite:1】``&#8203;.

## Model Training Overview
- **Shared Architecture:** Both tasks used similar BiLSTM structures with minor variations in embedding layers and hyperparameters.
- **Results:**
  - Task 1 (Validation): Precision - 82.14%, Recall - 73.91%, F1-Score - 77.81%.
  - Task 2 (Test): Precision - 89.37%, Recall - 92.80%, F1-Score - 91.05%.

## Bonus Task: Transformer Model
- **Data Preparation:** Followed similar processing as Task 1 with additional collate functions for transformer models.
- **Model Architecture:** Incorporated positional and token embedding classes as per assignment guidelines.
- **Purpose of Use:** This task aimed to explore the effectiveness of transformer models in comparison to BiLSTM models&#8203;``【oaicite:0】``&#8203;.

---

*Note: Detailed code and analysis are provided in the Jupyter notebooks attached.*
