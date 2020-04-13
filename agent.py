from sklearn.preprocessing import normalize
import numpy as np

class Agent:
    def __init__(self, anet, policy="probabilistic"):
        self.anet = anet
        self.policy = policy

    def best_action(self, valid_actions, state):
        if self.policy == "random":
            return np.random.choice(valid_actions)
        pred = self.anet.predict(state)[0]
        dist = [0 for _ in range(len(pred))]
        for i in valid_actions:
            dist[i] = pred[i]
        dist = normalize([dist], norm="l1")[0]
        if self.policy == "greedy":
            return np.argmax(dist)  # Index in dist represents an action
        return np.random.choice(range(len(pred)), p=dist)
