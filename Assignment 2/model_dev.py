from pathlib import Path
from xml.parsers.expat import errors
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from supervised_learning import LogisticRegression
from feature_selection import feature_hill_climb

BASE_DIR = Path(__file__).resolve().parent

# load historical customer data
customer_data = pd.read_csv(BASE_DIR / 'data' / 'customer_data.csv', index_col=None)
customer_claims = pd.read_csv(BASE_DIR / 'data' / 'customer_claims.csv', index_col=None)
data = pd.concat([customer_data, customer_claims[["claim"]]], axis=1)



# write your code to develop your agent program here.
# you can, of course, add additional .py files as needed.

def main():

    X = data.drop(columns=["claim"]).to_numpy(dtype=float)
    y = data["claim"].to_numpy(dtype=int)

    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        random_state=42,
        stratify=y_remaining
    )
    
    model = LogisticRegression(
        learning_rate=0.1,
        iterations=1000,
        regularization=0.01
    )

    print("\nSTARTING FEATURE SELECTION")
    print("--------------------------")

    selected_features, best_accuracy = feature_hill_climb(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    print("Training Logistic Regression...")

    model.gradient_descent(X_train, y_train)

    model_directory = BASE_DIR / "models"
    model_directory.mkdir(exist_ok=True)

    with open(model_directory / "logistic_model.pkl", "wb") as file:

        pickle.dump({
            "model": model,
            "feature_columns": list(data.drop(columns=["claim"]).columns)
        }, file)

    print("Model successfully saved.")

    with open(model_directory / "test_data.pkl", "wb") as file:

        pickle.dump((X_test, y_test), file)


if __name__ == "__main__":
    main()
