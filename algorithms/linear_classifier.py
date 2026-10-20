import numpy as np

class LinearClassifier:
    def __init__(self):
        self.w = None
        self.b = None

    def predict_scores(self, X):
        return np.asarray(X) @ self.w + self.b

    def predict(self, X):
        return np.where(self.predict_scores(X) >= 0, 1, -1)
