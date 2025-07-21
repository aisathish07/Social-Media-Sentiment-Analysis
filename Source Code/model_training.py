# src/model_training.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_and_evaluate_models(X_train, y_train, X_valid, y_valid):
    """
    Trains multiple models and returns their F1 scores.
    """
    models = {
        "Logistic Regression": LogisticRegression(random_state=17),
        "Decision Tree": DecisionTreeClassifier(random_state=17),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)
        
        accuracy = accuracy_score(y_valid, predictions)
        f1 = f1_score(y_valid, predictions)
        
        results[name] = f1
        print(f"{name} -> Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    # Create comparison DataFrame
    f1_scores = [results.get(m, 0) for m in models.keys()]
    compare_df = pd.DataFrame({
        'Model': list(models.keys()), 
        'F1_Score': f1_scores
    })
    
    print("\n--- Model Comparison ---")
    print(compare_df)
    
    # Plot comparison
    plt.figure(figsize=(8, 6))
    plt.bar(compare_df['Model'], compare_df['F1_Score'], color=['blue', 'green', 'orange'])
    plt.title('Comparison of Model Performance (F1 Score)')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.show()
    
    return compare_df