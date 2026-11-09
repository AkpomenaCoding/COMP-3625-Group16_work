
from pathlib import Path
import pickle

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    log_loss
)


BASE_DIR = Path(__file__).resolve().parent


def main():

    model_path = BASE_DIR / "models" / "logistic_model.pkl"
    test_path = BASE_DIR / "models" / "test_data.pkl"

    with open(model_path, "rb") as file:
        saved_data = pickle.load(file)

    with open(test_path, "rb") as file:
        X_test, y_test = pickle.load(file)

    model = saved_data["model"]

    probabilities = model.predict_probability(X_test)[:, 1]

    predictions = model.predict(X_test, threshold=0.5)

    print("\nMODEL EVALUATION")
    print("--------------------------")

    print(
        f"Accuracy: "
        f"{accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Brier Score: "
        f"{brier_score_loss(y_test, probabilities):.4f}"
    )

    print(
        f"Log Loss: "
        f"{log_loss(y_test, probabilities):.4f}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()
