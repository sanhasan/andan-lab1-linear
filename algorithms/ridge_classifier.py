import numpy as np

from algorithms.linear_classifier import LinearClassifier

# Алгоритм линейной регрессии в матричном виде с гребневой регуляризацией.
class RidgeRegressionClassifier(LinearClassifier):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_with_b = np.column_stack([np.ones(len(X)), X])

        I = np.eye(X_with_b.shape[1])
        I[0, 0] = 0

        theta = np.linalg.solve(
            X_with_b.T @ X_with_b + self.alpha * I,
            X_with_b.T @ y,
        )

        self.b = theta[0]
        self.w = theta[1:]
        return self
