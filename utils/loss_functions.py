import numpy as np


def logistic_loss(margin):
    return np.logaddexp(0, -margin)

def logistic_derivative(margin):
    return -1 / (1 + np.exp(margin))


def exponential_loss(margin):
    return np.exp(-margin)

def exponential_derivative(margin):
    return -np.exp(-margin)


def squared_hinge_loss(margin):
    return np.maximum(0, 1 - margin) ** 2

def squared_hinge_derivative(margin):
    return np.where(margin < 1, -2 * (1 - margin), 0)


LOSSES = {
    "logistic": (logistic_loss, logistic_derivative),
    "exponential": (exponential_loss, exponential_derivative),
    "squared_hinge": (squared_hinge_loss, squared_hinge_derivative),
}

# эмпирический риск, целевая функция, градиент
def loss_and_grad(theta, X, y, loss_name, l1, l2):
    loss_function, loss_derivative = LOSSES[loss_name]
    margin = y * (X @ theta)

    risk = np.mean(loss_function(margin))

    regularized_theta = theta.copy()
    regularized_theta[0] = 0

    objective = (
        risk
        + l1 * np.sum(np.abs(regularized_theta))
        + l2 / 2 * np.sum(regularized_theta**2)
    )

    gradient = (
        X.T @ (loss_derivative(margin) * y) / len(y) 
        + l1 * np.sign(regularized_theta) 
        + l2 * regularized_theta
    )

    return risk, objective, gradient
