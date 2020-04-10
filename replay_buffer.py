import numpy as np
import tensorflow as tf

class ReplayBuffer:
    def __init__(self):
        self.training_cases = []


    def get_random_minibatch(self, size):
        indices = np.random.randint(0, len(self.training_cases), min(size, len(self.training_cases)))
        out_x = []
        out_y = []
        for i in indices:
            out_x.append(self.training_cases[i][0])
            out_y.append(self.training_cases[i][1])
        return tf.convert_to_tensor(out_x, dtype="int32"), tf.convert_to_tensor(out_y, dtype="float32")

    def save(self, s, D):
        self.training_cases.append([s, D])