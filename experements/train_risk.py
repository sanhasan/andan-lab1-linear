from algorithms.gradient_classifier import GradientLinearClassifier


def run_train_risk(X, y, parameters, max_iter):
    model = GradientLinearClassifier(
        loss=parameters["best_loss"],
        learning_rate=parameters["best_learning_rate"],
        max_iter=max_iter,
        l1=parameters["best_l1"],
        l2=parameters["best_l2"],
    ).fit(X, y)
    return model.history_
