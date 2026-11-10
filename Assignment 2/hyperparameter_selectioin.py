import numpy as np
from supervised_learning import LogisticRegression
from sklearn.metrics import log_loss


def hyperparameter_search(
    X_train,
    y_train,
    X_validation,
    y_validation
):

    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]

    regularization_values = [0.001, 0.01, 0.1, 1.0]

    iteration_values = [200, 500, 1000, 1500, 2000]

    best_loss = float('inf')
    best_parameters = {}

    total_combinations = (
        len(learning_rates)
        * len(regularization_values)
        * len(iteration_values)
    )

    current_iteration = 0

    for learning_rate in learning_rates:

        for regularization in regularization_values:

            for iterations in iteration_values:

                current_iteration += 1

                print(
                    f"\nTesting configuration "
                    f"{current_iteration}/{total_combinations}"
                )

                model = LogisticRegression(
                    learning_rate=learning_rate,
                    iterations=iterations,
                    regularization=regularization
                )

                model.gradient_descent(
                    X_train,
                    y_train
                )

                probabilities = model.predict_probability(
                    X_validation
                )[:, 1]

                loss = log_loss(
                    y_validation,
                    probabilities,
                    labels=[0, 1]
                )

                print(f"Learning Rate: {learning_rate}")
                print(f"Regularization: {regularization}")
                print(f"Iterations: {iterations}")
                print(f"Validation Log Loss: {loss:.5f}")

                if loss < best_loss:

                    best_loss = loss

                    best_parameters = {
                        'learning_rate': learning_rate,
                        'regularization': regularization,
                        'iterations': iterations
                    }

                    print("NEW BEST PARAMETERS FOUND!")

    print("\nHYPERPARAMETER SEARCH COMPLETE")
    print("Best Parameters:", best_parameters)
    print("Best Validation Loss:", round(best_loss, 2))

    return best_parameters, best_loss
