# src/data_loader.py
import pandas as pd

def load_data():
    """
    Loads train and test data and combines them.
    """
    train_url = 'https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv'
    test_url = 'https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/test.csv'
    
    train_df = pd.read_csv(train_url)
    test_df = pd.read_csv(test_url)
    
    combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=True)
    
    return train_df, combined_df