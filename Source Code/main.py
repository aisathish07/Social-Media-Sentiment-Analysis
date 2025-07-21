# main.py
import warnings
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.data_preprocessing import preprocess_tweets
from src.eda import generate_wordclouds, analyze_hashtags
from src.feature_engineering import create_tfidf_features
from src.model_training import train_and_evaluate_models

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    """
    Runs the full sentiment analysis pipeline.
    """
    # 1. Load Data
    print("Step 1: Loading data...")
    train_df, combined_df = load_data()
    print("Data loaded successfully.")
    
    # 2. Preprocess Data
    print("\nStep 2: Preprocessing tweets...")
    combined_df = preprocess_tweets(combined_df)
    print("Preprocessing complete.")
    
    # 3. Exploratory Data Analysis
    print("\nStep 3: Performing EDA...")
    generate_wordclouds(combined_df)
    analyze_hashtags(combined_df)
    print("EDA complete. Plots are saved in the root directory.")
    
    # 4. Feature Engineering
    print("\nStep 4: Creating TF-IDF features...")
    tfidf_matrix, _ = create_tfidf_features(combined_df)
    print("Feature engineering complete.")
    
    # 5. Split Data for Training and Evaluation
    print("\nStep 5: Splitting data for modeling...")
    # The original train set had 31962 rows
    train_tfidf_matrix = tfidf_matrix[:31962]
    
    x_train, x_valid, y_train, y_valid = train_test_split(
        train_tfidf_matrix,
        train_df['label'],
        test_size=0.3,
        random_state=17
    )
    print("Data split complete.")
    
    # 6. Train and Evaluate Models
    print("\nStep 6: Training and evaluating models...")
    train_and_evaluate_models(x_train, y_train, x_valid, y_valid)
    print("\nPipeline finished.")

if __name__ == '__main__':
    main()