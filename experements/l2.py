import numpy as np

from algorithms.gradient_classifier import GradientLinearClassifier


def run_l2(X, y, parameters, values, max_iter):
    coefficients = []
    for value in values:
        model = GradientLinearClassifier(
            loss=parameters["best_loss"],
            learning_rate=parameters["best_learning_rate"],
            max_iter=max_iter,
            l1=0.0,
            l2=value,
            history_step=max_iter,
        ).fit(X, y)
        coefficients.append(model.w)

    return {
        "values": np.asarray(values),
        "coefficients": np.asarray(coefficients),
        "regularization": "l2",
    }
