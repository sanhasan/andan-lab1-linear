import numpy as np

from algorithms.linear_classifier import LinearClassifier
from algorithms.with_history import WithHistory
from utils.loss_functions import loss_and_grad


# алгоритм линейной классификации на основе градиентного спуска.
class GradientLinearClassifier(LinearClassifier, WithHistory):
    def __init__(self, loss="logistic", learning_rate=0.01, max_iter=1000, l1=0.0, l2=0.0, history_step=1):
        super().__init__()
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l1 = l1
        self.l2 = l2
        self.history_step = history_step
        self.history_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_with_b = np.column_stack([np.ones(len(X)), X])
        theta = np.zeros(X_with_b.shape[1])
        self.reset_history()

        for iteration in range(self.max_iter):
            risk, objective, gradient = loss_and_grad(
                theta, X_with_b, y, self.loss, self.l1, self.l2
            )

            if iteration % self.history_step == 0:
                self.save_history(iteration, risk, objective, theta)

            theta -= self.learning_rate * gradient

        risk, objective, _ = loss_and_grad(
            theta, X_with_b, y, self.loss, self.l1, self.l2
        )
        self.save_history(self.max_iter, risk, objective, theta)
        self.convert_history()

        self.b = theta[0]
        self.w = theta[1:]
        return self
