import numpy as np

from algorithms.ridge_classifier import RidgeRegressionClassifier
from experements.common import accuracy


def run_ridge_search(X_train, y_train, X_test, y_test, alphas):
    scores = []
    for alpha in alphas:
        model = RidgeRegressionClassifier(alpha=alpha).fit(X_train, y_train)
        scores.append(accuracy(y_test, model.predict(X_test)))

    scores = np.asarray(scores)
    best_index = np.argmax(scores)
    return {
        "alphas": np.asarray(alphas),
        "scores": scores,
        "best_alpha": alphas[best_index],
        "best_score": scores[best_index],
    }
