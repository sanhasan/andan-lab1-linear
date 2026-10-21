import numpy as np

from algorithms.gradient_classifier import GradientLinearClassifier
from experements.common import accuracy


def run_loss_search(
    X_train,
    y_train,
    X_test,
    y_test,
    losses,
    learning_rates,
    max_iter,
    history_step,
):
    loss_scores = np.zeros((len(losses), len(learning_rates)))
    for loss_index, loss in enumerate(losses):
        for rate_index, learning_rate in enumerate(learning_rates):
            model = GradientLinearClassifier(
                loss=loss,
                learning_rate=learning_rate,
                max_iter=max_iter,
                history_step=history_step,
            ).fit(X_train, y_train)
            loss_scores[loss_index, rate_index] = accuracy(y_test, model.predict(X_test))

    best_loss_index, best_rate_index = np.unravel_index(
        np.argmax(loss_scores), loss_scores.shape
    )
    best_loss = losses[best_loss_index]
    best_learning_rate = learning_rates[best_rate_index]

    return {
        "losses": list(losses),
        "learning_rates": np.asarray(learning_rates),
        "loss_scores": loss_scores,
        "best_loss": best_loss,
        "best_learning_rate": best_learning_rate,
        "best_loss_score": loss_scores[best_loss_index, best_rate_index],
    }

def run_elastic_net_search(
    X_train,
    y_train,
    X_test,
    y_test,
    loss,
    learning_rate,
    l1_values,
    l2_values,
    max_iter,
    history_step,
):
    regularization_scores = np.zeros((len(l1_values), len(l2_values)))
    for l1_index, l1 in enumerate(l1_values):
        for l2_index, l2 in enumerate(l2_values):
            model = GradientLinearClassifier(
                loss=loss,
                learning_rate=learning_rate,
                max_iter=max_iter,
                l1=l1,
                l2=l2,
                history_step=history_step,
            ).fit(X_train, y_train)
            regularization_scores[l1_index, l2_index] = accuracy(
                y_test, model.predict(X_test)
            )

    best_l1_index, best_l2_index = np.unravel_index(
        np.argmax(regularization_scores), regularization_scores.shape
    )
    return {
        "l1_values": np.asarray(l1_values),
        "l2_values": np.asarray(l2_values),
        "regularization_scores": regularization_scores,
        "best_l1": l1_values[best_l1_index],
        "best_l2": l2_values[best_l2_index],
        "best_score": regularization_scores[best_l1_index, best_l2_index],
    }
