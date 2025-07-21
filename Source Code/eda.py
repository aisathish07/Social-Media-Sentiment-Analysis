# src/eda.py
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import requests

def generate_wordclouds(df):
    """Generates and saves word clouds for positive and negative tweets."""
    print("Generating word clouds...")
    # Get mask image
    try:
        mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
    except Exception as e:
        print(f"Could not download mask image, continuing without it. Error: {e}")
        mask = None

    # Positive word cloud
    positive_words = ' '.join(text for text in df['Cleaned_Tweets'][df['label'] == 0])
    wc_pos = WordCloud(background_color='black', height=1500, width=4000, mask=mask).generate(positive_words)
    plt.figure(figsize=(10, 8))
    plt.imshow(wc_pos, interpolation="bilinear")
    plt.axis('off')
    plt.title('Positive Words Word Cloud')
    plt.savefig('positive_wordcloud.png')
    plt.close()

    # Negative word cloud
    negative_words = ' '.join(text for text in df['Cleaned_Tweets'][df['label'] == 1])
    wc_neg = WordCloud(background_color='black', height=1500, width=4000, mask=mask).generate(negative_words)
    plt.figure(figsize=(10, 8))
    plt.imshow(wc_neg, interpolation="bilinear")
    plt.axis('off')
    plt.title('Negative Words Word Cloud')
    plt.savefig('negative_wordcloud.png')
    plt.close()
    print("Word clouds saved.")

def analyze_hashtags(df):
    """Extracts, counts, and plots the most frequent hashtags."""
    print("Analyzing hashtags...")
    def extract_hashtags(x):
        hashtags = []
        for i in x:
            ht = re.findall(r'#(\w+)', i)
            hashtags.append(ht)
        return sum(hashtags, [])

    # Positive hashtags
    pos_hashtags = extract_hashtags(df['Cleaned_Tweets'][df['label'] == 0])
    pos_freq = nltk.FreqDist(pos_hashtags)
    pos_df = pd.DataFrame({'Hashtags': list(pos_freq.keys()), 'Count': list(pos_freq.values())})
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=pos_df.nlargest(20, columns='Count'), y='Hashtags', x='Count')
    plt.title('Top 20 Positive Hashtags')
    sns.despine()
    plt.savefig('positive_hashtags.png')
    plt.close()

    # Negative hashtags
    neg_hashtags = extract_hashtags(df['Cleaned_Tweets'][df['label'] == 1])
    neg_freq = nltk.FreqDist(neg_hashtags)
    neg_df = pd.DataFrame({'Hashtags': list(neg_freq.keys()), 'Count': list(neg_freq.values())})

    plt.figure(figsize=(10, 8))
    sns.barplot(data=neg_df.nlargest(20, columns='Count'), y='Hashtags', x='Count')
    plt.title('Top 20 Negative Hashtags')
    sns.despine()
    plt.savefig('negative_hashtags.png')
    plt.close()
    print("Hashtag analysis plots saved.")