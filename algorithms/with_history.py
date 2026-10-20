import numpy as np

class WithHistory:
    def reset_history(self):
        self.history_ = {
            "iteration": [],
            "risk": [],
            "objective": [],
            "weights": [],
            "bias": [],
        }

    def save_history(self, iteration, risk, objective, theta):
        self.history_["iteration"].append(iteration)
        self.history_["risk"].append(risk)
        self.history_["objective"].append(objective)
        self.history_["weights"].append(theta[1:].copy())
        self.history_["bias"].append(theta[0])

    def convert_history(self):
        for name in self.history_:
            self.history_[name] = np.asarray(self.history_[name])
