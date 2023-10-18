# PYTHON VERSION: Python 3.9.12

# ! pip install bs4 # in case you don't have it installed
# ! pip install nltk
# ! pip install scikit-learn
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
import re
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

######################################################################## Defining Functions #####################################################################

def extract_text_from_html(text):
    # Removing <style> and <script> tags and their content
    text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)

    # Removing <a> tags, keeping their inner text
    text = re.sub(r'<a.*?>(.*?)</a>', r'\1', text)

    # Removing any remaining HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Joining the stripped strings
    text = ' '.join(text.strip().split())
    return text

def contract_text(text):
    contractions_dict = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }
    
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)
        
    return text

def clean_text(text):
    # len_before_cleaning = len(text)
    text = text.lower()
    text = extract_text_from_html(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    text = contract_text(text)
    # len_after_cleaning = len(text)
    return text

def process_row(row):
    cleaned_text = clean_text(row['review_body'])
    row['review_body'] = cleaned_text
    return row

def filter_out_stopwords(input_text):
    """Filters out English stop words from a text."""
    english_stops = set(stopwords.words('english'))
    tokenized_words = word_tokenize(input_text)

    filtered_words = []
    for word in tokenized_words:
        if word.lower() not in english_stops:
            filtered_words.append(word)

    return ' '.join(filtered_words)

def get_wordnet_pos(treebank_tag):
    pos_mapping = {
        'J': ADJ,
        'V': VERB,
        'N': NOUN,
        'R': ADV
    }
    return pos_mapping.get(treebank_tag[0], NOUN)

def lemmatize(text):
    """lemmatizes the words of a sentence"""
    result = []
    text = nltk.pos_tag(word_tokenize(text))
    lem = WordNetLemmatizer()
    for word in text:
        result.append(lem.lemmatize(word[0],get_wordnet_pos(word[-1])))
    return ' '.join(result)

def remove_stop_word_and_lemmatize(text):
    text = filter_out_stopwords(text)
    text = lemmatize(text)
    return text

def stop_word_lemmatization(row):
    cleaned_text = remove_stop_word_and_lemmatize(row['review_body'])
    row['review_body'] = cleaned_text
    return row

######################################################################## Starting Code #####################################################################
df = pd.read_table('data.tsv',on_bad_lines='skip')
data = df[['star_rating','review_body']] #keeping only columns needed
data = data.dropna(axis=0)
data['star_rating']=data['star_rating'].astype('int')
class1 = data[data['star_rating']<=3].copy() #defining class 1 for ratings with values 1,2,3
labels = [1]*len(class1)
class1.loc[:,'label'] = labels

class2 = data[data['star_rating']>=4].copy()  #defining class 2 for ratings with values 4,5
labels = [2]*len(class2)
class2.loc[:,'label'] = labels

sampled_class1 = class1.sample(n=50000, random_state=42)  # Using a fixed random state for reproducibility
sampled_class2 = class2.sample(n=50000, random_state=42)
# Concatenating the sampled data to create a balanced dataset
balanced_data = pd.concat([sampled_class1, sampled_class2], ignore_index=True)
# Shuffle the dataset
training_dataset = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

######################################################################## Data Cleaning #####################################################################

average_length_before = training_dataset['review_body'].apply(len).mean()
training_dataset = training_dataset.apply(process_row, axis=1, result_type='expand')
average_length_after =training_dataset['review_body'].apply(len).mean()
print(f"{average_length_before},{average_length_after}")

######################################################################## Stemming and Lemmatization #####################################################################

average_length_before = training_dataset['review_body'].apply(len).mean()
training_dataset = training_dataset.apply(stop_word_lemmatization, axis=1, result_type='expand')
average_length_after =training_dataset['review_body'].apply(len).mean()
print(f"{average_length_before},{average_length_after}")

############################################################### TF IDF, BOW and train test split #####################################################

vectorizer = TfidfVectorizer()
frequency_matrix = vectorizer.fit_transform(training_dataset['review_body'])

count_vectorizer = CountVectorizer()
bow_features = count_vectorizer.fit_transform(training_dataset['review_body'])

X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(frequency_matrix, training_dataset['label'], test_size=0.2, random_state=42)
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow_features, training_dataset['label'], test_size=0.2, random_state=42)


######################################################################## Perceptron ####################################################################
# param_grid = {
#     'eta0': [1e-6,5e-6,3e-6,2e-6,0.00001,0.0003], #0.0001,0.0005, 0.001
#     'max_iter': [10000,50000],
#     'penalty': [None, 'l2', 'l1', 'elasticnet'],
#     'alpha': [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01,0.05,0.1]
# }
param_grid = {
    'eta0': [2e-6,3e-6,6e-6,7e-6,7e-7,7e-10,8e-7,0.00001], 
    'max_iter': [700,1000,2000],#5000,10000
    'penalty': [None, 'l2', 'l1', 'elasticnet'],
    'alpha': [0.012,0.0131,0.0132,0.0133,0.0134,0.135,0.014,0.015] 
}
grid= GridSearchCV(Perceptron(), param_grid, scoring = 'f1')
grid.fit(X_train_bow,y_train_bow)
best_percep = grid.best_estimator_
percep_preds_bow = best_percep.predict(X_test_bow)

precision = precision_score(y_test_bow, percep_preds_bow)  
recall = recall_score(y_test_bow, percep_preds_bow) 
f1 = f1_score(y_test_bow, percep_preds_bow)
print(f"{precision} {recall} {f1}")

param_grid = {
    'eta0': [0.05,0.03,0.01, 0.001,0.003],
    'max_iter': [5000,10000], #10000,
    'l1_ratio':[0,0.02,0.05, 0.1,0.125,0.15,0.175,0.25],
    'penalty': ['elasticnet'], #'l2', 'l1',
    'alpha': [1e-6,5e-6,1e-5,5e-5]
}
grid= GridSearchCV(Perceptron(), param_grid, scoring = 'f1')
grid.fit(X_train_tf,y_train_tf)
best_percep = grid.best_estimator_
percep_preds_tf = best_percep.predict(X_test_tf)

precision = precision_score(y_test_tf, percep_preds_tf)  
recall = recall_score(y_test_tf, percep_preds_tf) 
f1 = f1_score(y_test_tf, percep_preds_tf)
print(f"{precision} {recall} {f1}")

######################################################################## SVM ####################################################################
param_grid= {
    'C': [0.001, 0.005, 0.01, 0.012, 0.015, 0.017, 0.02, 0.025],
    'tol': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
    'max_iter': [1000000],
    'intercept_scaling': [0.9, 0.95, 1.0, 1.05, 1.1]
}
svm = LinearSVC()
grid_search = GridSearchCV(svm, param_grid, scoring='f1', cv=5) 
grid_search .fit(X_train_bow, y_train_bow)
best_svm = grid_search.best_estimator_
svm_preds_bow = best_svm.predict(X_test_bow)
precision = precision_score(y_test_bow, svm_preds_bow)  
recall = recall_score(y_test_bow, svm_preds_bow) 
f1 = f1_score(y_test_bow, svm_preds_bow)
print(f"{precision} {recall} {f1}")

param_grid = {
    'C': [0.001, 0.005, 0.015, 0.025],
    'max_iter': [1000000],
    'tol': [0.00001],
    'intercept_scaling': [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
}
svm = LinearSVC()
grid_search = GridSearchCV(svm, param_grid, scoring='f1', cv=5)  # Using 5-fold cross-validation
grid_search .fit(X_train_tf, y_train_tf)
best_svm = grid_search.best_estimator_
svm_preds_tf = best_svm.predict(X_test_tf)
precision = precision_score(y_test_tf, svm_preds_tf)  
recall = recall_score(y_test_tf, svm_preds_tf) 
f1 = f1_score(y_test_tf, svm_preds_tf)
print(f"{precision} {recall} {f1}")

######################################################################## LR ####################################################################
param_grid = {
    'C': [0.1, 0.25, 0.5, 1.0],
    'solver': ['lbfgs','liblinear','saga'],
    'max_iter': [10000,100000]
}
grid_search = GridSearchCV(LogisticRegression(), param_grid, scoring='f1')
grid_search.fit(X_train_bow, y_train_bow)
best_lr = grid_search.best_estimator_
lr_preds_bow = best_lr.predict(X_test_bow)
precision = precision_score(y_test_bow, lr_preds_bow)  
recall = recall_score(y_test_bow, lr_preds_bow) 
f1 = f1_score(y_test_bow, lr_preds_bow)
print(f"{precision} {recall} {f1}")

param_grid = {
    'C': [0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    'solver': ['lbfgs','liblinear','saga'],
    'max_iter': [10000,100000]
}
grid_search = GridSearchCV(LogisticRegression(), param_grid, scoring='f1')
grid_search.fit(X_train_tf, y_train_tf)
best_lr = grid_search.best_estimator_
lr_preds_tf = best_lr.predict(X_test_tf)
precision = precision_score(y_test_tf, lr_preds_tf)  
recall = recall_score(y_test_tf, lr_preds_tf) 
f1 = f1_score(y_test_tf, lr_preds_tf)
print(f"{precision} {recall} {f1}")



######################################################################## NB ####################################################################

param_grid = { 'alpha': [1.0,2.5,5.5,5.75,5.95,6.0,6.15,6.25, 6.5,10.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, scoring='f1',cv=10)
grid_search.fit(X_train_bow, y_train_bow)
best_mnb = grid_search.best_estimator_
mnb_preds_bow = best_mnb.predict(X_test_bow)
precision = precision_score(y_test_bow, mnb_preds_bow)  
recall = recall_score(y_test_bow, mnb_preds_bow) 
f1 = f1_score(y_test_bow, mnb_preds_bow)
print(f"{precision} {recall} {f1}")


param_grid = { 'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0,100.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, scoring='f1',cv=10)
grid_search.fit(X_train_tf, y_train_tf)
best_mnb = grid_search.best_estimator_
mnb_preds_tf = best_mnb.predict(X_test_tf)
precision = precision_score(y_test_tf, mnb_preds_tf)  
recall = recall_score(y_test_tf, mnb_preds_tf) 
f1 = f1_score(y_test_tf, mnb_preds_tf)
print(f"{precision} {recall} {f1}")


