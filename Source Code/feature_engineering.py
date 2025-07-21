# src/feature_engineering.py
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(df):
    """
    Creates TF-IDF features from the cleaned tweets.
    """
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.90, min_df=2, max_features=1000, stop_words='english'
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Cleaned_Tweets'])
    return tfidf_matrix, tfidf_vectorizer