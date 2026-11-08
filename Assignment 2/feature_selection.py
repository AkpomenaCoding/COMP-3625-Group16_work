import pandas as pd
import numpy as np
from supervised_learning import LogisticRegression
from sklearn.metrics import accuracy_score

def feature_hill_climb(X_train, y_train, X_validation, y_validation):
    
    selected_features = list()
    unselected_features = set(X_train.columns)
    best_acc = -1
    
    while unselected_features:
        
        improved = False
        best_test_acc = best_acc
        best_test_feature = next(iter(unselected_features))
        
        for feature in unselected_features:
            current_features = selected_features.copy() + [feature]
            
            X_train_subset = X_train[current_features]
            X_validation_subset = X_validation[current_features]
            
            model = LogisticRegression()
            model.gradient_descent(X_train_subset, y_train)
            predictions = model.predict_class(X_validation_subset)
            acc = accuracy_score(y_validation, predictions)
            
            if acc > best_test_acc:
                best_test_acc = acc
                best_test_feature = feature
                improved = True
        
        if not improved:
            break
        
        selected_features.append(best_test_feature)
        unselected_features.remove(best_test_feature)
        best_acc = best_test_acc
            
    return list(selected_features), best_acc