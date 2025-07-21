# src/data_preprocessing.py
import re
import numpy as np
from nltk.stem import PorterStemmer

def remove_pattern(text, pattern):
    """Removes a regex pattern from text."""
    r = re.findall(pattern, text)
    for i in r:
        text = re.sub(i, "", text)
    return text

def preprocess_tweets(df):
    """
    Cleans and preprocesses the tweet text in a DataFrame.
    """
    # Remove user mentions
    df['Cleaned_Tweets'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
    
    # Remove special characters, numbers, and punctuation (keeping hashtags)
    df['Cleaned_Tweets'] = df['Cleaned_Tweets'].str.replace("[^a-zA-Z#]", " ")
    
    # Remove short words (length < 4)
    df['Cleaned_Tweets'] = df['Cleaned_Tweets'].apply(
        lambda x: ' '.join([w for w in x.split() if len(w) > 3])
    )
    
    # Tokenize and stem
    tokenized_tweets = df['Cleaned_Tweets'].apply(lambda x: x.split())
    ps = PorterStemmer()
    tokenized_tweets = tokenized_tweets.apply(lambda x: [ps.stem(i) for i in x])
    
    # Re-join tokens into a single string
    for i in range(len(tokenized_tweets)):
        tokenized_tweets[i] = ' '.join(tokenized_tweets[i])
        
    df['Cleaned_Tweets'] = tokenized_tweets
    
    return df
