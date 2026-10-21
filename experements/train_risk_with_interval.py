import numpy as np
from scipy.stats import t

from algorithms.gradient_classifier import GradientLinearClassifier
from algorithms.ridge_classifier import RidgeRegressionClassifier
from experements.common import accuracy
from utils.normalization import prepare_features, split_dataset


def accuracy_history(model, X, y):
    scores = X @ model.history_["weights"].T + model.history_["bias"]
    predictions = np.where(scores >= 0, 1, -1)
    return np.mean(predictions == y[:, np.newaxis], axis=0)


def collect_learning_curve_data(
    X,
    y,
    gradient_parameters,
    ridge_alpha,
    features_count,
    runs_count,
    test_size,
    random_state,
    max_iter,
    history_step,
    confidence,
):
    gradient_scores = []
    ridge_scores = []
    iterations = None

    for run in range(runs_count):
        X_train, X_test, y_train, y_test = split_dataset(
            X,
            y,
            test_size,
            random_state + run,
        )
        X_train, X_test = prepare_features(
            X_train,
            X_test,
            y_train,
            features_count,
        )

        gradient_model = GradientLinearClassifier(
            loss=gradient_parameters["best_loss"],
            learning_rate=gradient_parameters["best_learning_rate"],
            max_iter=max_iter,
            l1=gradient_parameters["best_l1"],
            l2=gradient_parameters["best_l2"],
            history_step=history_step,
        ).fit(X_train, y_train)

        gradient_scores.append(accuracy_history(gradient_model, X_test, y_test))
        iterations = gradient_model.history_["iteration"]

        ridge_model = RidgeRegressionClassifier(alpha=ridge_alpha).fit(
            X_train, y_train
        )
        ridge_scores.append(accuracy(y_test, ridge_model.predict(X_test)))

    data = {
        "iterations": iterations,
        "gradient_scores": np.asarray(gradient_scores),
        "ridge_scores": np.asarray(ridge_scores),
    }
    return calculate_confidence_intervals(data, confidence)


def calculate_confidence_intervals(data, confidence):
    gradient_scores = data["gradient_scores"]
    ridge_scores = data["ridge_scores"]
    runs_count = len(ridge_scores)
    critical_value = t.ppf((1 + confidence) / 2, runs_count - 1) # критическое значение распределения Стьюдента

    gradient_mean = gradient_scores.mean(axis=0)
    ridge_mean = ridge_scores.mean()

    # ddof - поправку к количеству степеней свободы (на сколько понял, если у нас не вся выборка, тогда надо учесть это и уменьшить коэф)
    gradient_interval = critical_value * gradient_scores.std(axis=0, ddof=1) / np.sqrt(runs_count)
    ridge_interval = critical_value * ridge_scores.std(ddof=1) / np.sqrt(runs_count)

    return {
        **data,
        "confidence": confidence,
        "gradient_mean": gradient_mean,
        "gradient_lower": gradient_mean - gradient_interval,
        "gradient_upper": gradient_mean + gradient_interval,
        "ridge_mean": ridge_mean,
        "ridge_lower": ridge_mean - ridge_interval,
        "ridge_upper": ridge_mean + ridge_interval,
    }
