import numpy as np

from algorithms.ridge_classifier import RidgeRegressionClassifier
from experements.common import accuracy


def run_ridge_search(X_train, y_train, X_test, y_test, alphas):
    scores = []
    scores_train = []
    for alpha in alphas:
        model = RidgeRegressionClassifier(alpha=alpha).fit(X_train, y_train)
        scores.append(accuracy(y_test, model.predict(X_test)))
        scores_train.append(accuracy(y_train, model.predict(X_train)))

    scores = np.asarray(scores)
    best_index = np.argmax(scores)

    scores_train = np.asarray(scores_train)
    best_index_train = np.argmax(scores_train)
    return {
        "alphas": np.asarray(alphas),
        "scores": scores,
        "best_alpha": alphas[best_index],
        "best_score": scores[best_index],
        "best_alpha_train": alphas[best_index_train],
        "best_score_train": scores_train[best_index_train],
    }
