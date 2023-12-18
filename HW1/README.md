# CSCI 544: Assignment 1 Report

## Grade: 100/100

## 1. Data Acquisition and Initial Handling
The dataset was sourced from a provided link and stored locally after extraction. Key columns focused on were 'Reviews' and 'Ratings', referred to as 'review_body' and 'star_ratings', respectively. These columns were isolated for further processing.

## 2. Preliminary Data Cleansing
Initial data cleaning involved setting 'star_ratings' to a uniform integer type. A new column ‘label’ was introduced for categorization (1 for ratings ≤3, 2 for ratings ≥4), forming a dataset of 100,000 rows after segregation and merging. 

### Textual Cleaning
- Conversion to lowercase.
- Removal of HTML elements/URLs using BeautifulSoup.
- Filtering out non-letter characters using regex.
- Applying text contractions with a dictionary of English contractions.

`clean_text()` function was used on each entry in 'review_body' using pandas’ apply() function. Average text lengths before and after cleaning were compared.

## 3. Stemming and Lemmatization
Utilizing NLTK, the data underwent:
- Tokenization.
- Removal of stopwords.
- POS tagging.
- Lemmatization on tagged data.

## 4. Data Preparation for Modeling
Post-processing, features were extracted:
- Tf-IDF features.
- Bag Of Words (BOW) features.

The dataset was split into 80% training and 20% testing sets. Various models were trained using sklearn, and outcomes were measured in terms of precision, recall, and F1 scores. Models included Perceptron, SVM, Logistic Regression, and Multinomial Naive Bayes, each with both Tf-IDF and BOW features.

### Key Results
- Perceptron with Tf-IDF achieved a precision of 0.7889, recall of 0.8119, and F1 score of 0.8002.
- SVM with BOW achieved a precision of 0.8560, recall of 0.8187, and F1 score of 0.8370.
- Logistic Regression with Tf-IDF achieved a precision of 0.8373, recall of 0.8500, and F1 score of 0.8436.
- Multinomial Naive Bayes with Tf-IDF achieved a precision of 0.7884, recall of 0.8548, and F1 score of 0.8203.

---

*Note: This report is a concise summary of the assignment. For detailed code and analysis, refer to the Jupyter notebooks attached.*
