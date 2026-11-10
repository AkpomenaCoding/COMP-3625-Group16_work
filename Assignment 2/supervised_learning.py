import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.1, iterations=1000,
                 regularization=0.01):

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization

        self.weights = None
        self.bias = 0

        self.X_mean = None
        self.X_std = None

    def normalize_feature(self, X):

        return (X - self.X_mean) / self.X_std

    def sigmoid(self, weight):

        weight = np.clip(weight, -500, 500)

        return 1 / (1 + np.exp(-weight))

    def check_claim_probability(self, X):

        weights = X @ self.weights + self.bias

        return self.sigmoid(weights)

    def cross_entropy_loss(self, y_true, y_pred):

        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

        loss = -np.mean(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

        penalty = (
            self.regularization
            / (2 * len(y_true))
            * np.sum(self.weights ** 2)
        )

        return loss + penalty

    def gradient_descent(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)

        self.X_std[self.X_std == 0] = 1

        X = self.normalize_feature(X)

        number_of_samples, number_of_features = X.shape

        self.weights = np.zeros(number_of_features)
        self.bias = 0.0

        for i in range(self.iterations):

            predictions = self.check_claim_probability(X)

            errors = predictions - y

            weight_gradient = (
                X.T @ errors / number_of_samples
                + self.regularization
                / number_of_samples * self.weights
            )

            bias_gradient = np.sum(errors) / number_of_samples

            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

            if (i + 1) % 100 == 0:

                updated_predictions = self.check_claim_probability(X)
                
                loss = self.cross_entropy_loss(y, updated_predictions)

                print(
                    f"Iteration {i + 1}/{self.iterations}, "
                    f"Loss: {loss:.4f}"
                )

        return self

    def predict_probability(self, X):

        X = np.asarray(X, dtype=float)

        X = self.normalize_feature(X)

        probabilities = self.check_claim_probability(X)

        return np.column_stack((
            1 - probabilities,
            probabilities
        ))

    def predict_class(self, X, threshold=0.5):

        probabilities = self.predict_probability(X)[:, 1]

        predictions = []

        for probability in probabilities:
            if probability >= threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions
    
    