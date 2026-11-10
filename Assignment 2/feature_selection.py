import pandas as pd
import numpy as np
from supervised_learning import LogisticRegression
from sklearn.metrics import log_loss

def feature_hill_climb(X_train, y_train, X_validation, y_validation):
    
    selected_features = []
    unselected_features = set(X_train.columns)
    best_loss = float('inf')
    iteration = 0

    while unselected_features:
        
        improved = False
        best_test_loss = best_loss
        best_test_feature = next(iter(unselected_features))
                
        iteration += 1

        print(f"\nFeature Selection Iteration: {iteration}")
        print(f"Selected Features: {selected_features}")
        print(f"Remaining Features: {len(unselected_features)}")

        
        for feature in unselected_features:
            print(f"Testing feature: {feature}")

            current_features = selected_features.copy() + [feature]
            
            X_train_subset = X_train[current_features].to_numpy()
            X_validation_subset = X_validation[current_features].to_numpy()
            
            model = LogisticRegression(learning_rate=0.1, iterations=200, regularization=0.01)
            model.gradient_descent(X_train_subset, y_train)
            probabilities = model.predict_probability(X_validation_subset)[:, 1]
            loss = log_loss(y_validation, probabilities)

            print(f"Validation Log Loss: {loss:.4f}")

            if loss < best_test_loss:
                best_test_loss = loss
                best_test_feature = feature
                improved = True

        
        if not improved:
            break
        
        selected_features.append(best_test_feature)
        unselected_features.remove(best_test_feature)
        best_loss = best_test_loss
            
    return list(selected_features), best_loss