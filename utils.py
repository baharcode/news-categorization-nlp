# utils.py

import yaml
import pandas as pd
import re
import emoji
import logging
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Make sure to download these resources if not already done
nltk.download('stopwords')
nltk.download('wordnet')

# Load configuration from YAML file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define custom transformer to select columns
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logging.info(f"Selecting column: {self.column}")
        return X[self.column]

# Text preprocessor
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        logging.info("Starting text preprocessing...")
        X['title_cleaned'] = X['title'].apply(lambda text: self.clean_text(text) if pd.notna(text) else '')
        X['body_cleaned'] = X['body'].apply(lambda text: self.clean_text(text) if pd.notna(text) else '')
        logging.info("Text preprocessing completed.")
        return X

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'http://\S+|https://\S+|www\.\S+', '', text, flags=re.MULTILINE)
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d', ' ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

# Custom transformer for title TF-IDF calculation
class TitleTfidfTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tfidf_params=None):
        self.tfidf_vectorizer = TfidfVectorizer(**tfidf_params)

    def fit(self, X, y=None):
        logging.info("Fitting TitleTfidfTransformer...")
        self.tfidf_vectorizer.fit(X)
        return self

    def transform(self, X):
        logging.info("Transforming title text to TF-IDF features...")
        return self.tfidf_vectorizer.transform(X)

# Custom transformer for body TF-IDF calculation with weight
class BodyTfidfTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, weight=1.45, tfidf_params=None):
        self.weight = weight
        self.tfidf_vectorizer = TfidfVectorizer(**tfidf_params)

    def fit(self, X, y=None):
        logging.info("Fitting BodyTfidfTransformer...")
        self.tfidf_vectorizer.fit(X)
        return self

    def transform(self, X):
        logging.info("Transforming body text to TF-IDF features...")
        tfidf_matrix = self.tfidf_vectorizer.transform(X)
        return tfidf_matrix * self.weight
