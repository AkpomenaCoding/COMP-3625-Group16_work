from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from supervised_learning import LogisticRegression
from feature_selection import feature_hill_climb
from hyperparameter_selectioin import hyperparameter_search

BASE_DIR = Path(__file__).resolve().parent

# load historical customer data
customer_data = pd.read_csv(BASE_DIR / 'data' / 'customer_data.csv', index_col=None)
customer_claims = pd.read_csv(BASE_DIR / 'data' / 'customer_claims.csv', index_col=None)
data = pd.concat([customer_data, customer_claims[["claim"]]], axis=1)



# write your code to develop your agent program here.
# you can, of course, add additional .py files as needed.

def main():

    X = data.drop(columns=["claim"])
    y = data["claim"].to_numpy()

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

    selected_features, best_loss = feature_hill_climb(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    print(f"\nSELECTED FEATURES: {selected_features}")
    print(f"BEST LOSS: {best_loss:.4f}")

    print("Training Logistic Regression")

    X_train_selected = X_train[selected_features].to_numpy()
    X_test_selected = X_test[selected_features].to_numpy()
    
    X_validation_selected = X_validation[selected_features].to_numpy()
    
    best_parameters, best_loss = hyperparameter_search(
        X_train_selected,
        y_train,
        X_validation_selected,
        y_validation
    )

    validation_model = LogisticRegression(
        learning_rate=best_parameters['learning_rate'],
        iterations=best_parameters['iterations'],
        regularization=best_parameters['regularization']
    )

    validation_model.gradient_descent(
        X_train_selected,
        y_train
    )

    probabilities = validation_model.predict_probability(X_validation_selected)[:, 1]
        
    best_threshold = 0.5
    best_f1 = -1
        
    for threshold in np.arange(0.05, 0.55, 0.01):
        
        predictions = (probabilities >= threshold).astype(int)
        
        recall = recall_score(y_validation, predictions)
        precision = precision_score(y_validation, predictions)
        f1 =    f1_score(y_validation, predictions)
        
        print(
            f"Threshold: {threshold:.2f} | "
            f"Recall: {recall:.4f} | "
            f"Precision: {precision:.4f} | "
            f"F1: {f1:.4f}"
            )
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = round(threshold, 2)
        
    print("\nBest threshold:", best_threshold)
    
    X_train_final = pd.concat([
        X_train[selected_features],
        X_validation[selected_features]
    ]).to_numpy()

    y_train_final = np.concatenate([
        y_train,
        y_validation
    ])

    final_model = LogisticRegression(
    learning_rate=best_parameters['learning_rate'],
    iterations=best_parameters['iterations'],
    regularization=best_parameters['regularization']
    )

    final_model.gradient_descent(
        X_train_final,
        y_train_final
    )
    

    model_directory = BASE_DIR / "models"
    model_directory.mkdir(exist_ok=True)

    with open(model_directory / "logistic_model.pkl", "wb") as file:

        pickle.dump({
            "model": final_model,
            "feature_columns": selected_features,
            "threshold": best_threshold
        }, file)

    print("Model successfully saved.")

    with open(model_directory / "test_data.pkl", "wb") as file:

        pickle.dump((X_test_selected, y_test), file)


if __name__ == "__main__":
    main()
