import pickle
import random

import torch

from config import training

class ReplayBuffer:
    def __init__(self):
        self.training_cases = []


    def get_random_minibatch(self, batch_size):
        out = random.sample(self.training_cases, min(batch_size, len(self.training_cases)))
        # print("asdf", [o[1] for o in out])
        return torch.tensor([o[0] for o in out], dtype=torch.float), torch.tensor([o[1] for o in out], dtype=torch.float)

    def save(self, s, D):
        self.training_cases.append([s, D])
        if len(self.training_cases) > training["buffer max size"]:
            self.training_cases.pop(0)

    def save_to_file(self):
        pickle.dump(self.training_cases, open(f"buffers/{training['file prefix']}", "wb"))
