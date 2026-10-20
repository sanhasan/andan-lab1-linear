import numpy as np


def select_features(X_train, y_train, features_count):
    X_centered = X_train - X_train.mean(axis=0)
    y_centered = y_train - y_train.mean()
    numerator = X_centered.T @ y_centered
    denominator = np.sqrt(
        np.sum(X_centered**2, axis=0) * np.sum(y_centered**2)
    )
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros(X_train.shape[1], dtype=np.float64),
        where=denominator > 0,
    )
    return np.argsort(-np.abs(correlations))[:features_count]
