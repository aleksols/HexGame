from sklearn.preprocessing import normalize
import numpy as np

class Agent:
    def __init__(self, anet, policy="probabilistic", name=""):
        self.anet = anet
        self.policy = policy
        self.name = name

    def best_action(self, valid_actions, state, verbose=False):
        if self.policy == "random":
            return np.random.choice(valid_actions)
        pred = self.anet.predict(state)
        dist = [0 for _ in range(len(pred))]
        for i in valid_actions:
            dist[i] = pred[i]
        dist = normalize([dist], norm="l1")[0]
        if verbose:
            print("dist", self, dist)
            print("max", np.argmax(dist))
        if self.policy == "greedy":
            return np.argmax(dist)  # Index in dist represents an action
        return np.random.choice(range(len(pred)), p=dist)
