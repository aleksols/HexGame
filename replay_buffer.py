import numpy as np
import tensorflow as tf

class ReplayBuffer:
    def __init__(self):
        self.training_cases = []


    def get_random_minibatch(self, size):
        indices = np.random.randint(0, len(self.training_cases), min(size, len(self.training_cases)))
        print("Indices. Check that they are unique", indices)
        out = []
        for i in indices:
            out.append(self.training_cases[i])
        return out

    def save(self, s, D):
        self.training_cases.append([s, D])