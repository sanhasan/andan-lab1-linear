import numpy as np

from algorithms.gradient_classifier import GradientLinearClassifier


def run_l1(X, y, parameters, values, max_iter):
    coefficients = []
    for value in values:
        model = GradientLinearClassifier(
            loss=parameters["best_loss"],
            learning_rate=parameters["best_learning_rate"],
            max_iter=max_iter,
            l1=value,
            l2=0.0,
            history_step=max_iter,
        ).fit(X, y)
        coefficients.append(model.w)

    return {
        "values": np.asarray(values),
        "coefficients": np.asarray(coefficients),
        "regularization": "l1",
    }
