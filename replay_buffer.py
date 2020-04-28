import pickle
import random

import torch

from config import TrainingConf

class ReplayBuffer:
    def __init__(self, predict_value, conv_input):
        self.training_cases = []
        self.include_value = predict_value
        self.conv_input = conv_input

    def get_random_minibatch(self, batch_size):
        out = random.sample(self.training_cases, min(batch_size, len(self.training_cases)))
        # print("asdf", [o[1] for o in out])
        if self.include_value and self.conv_input:
            features = torch.stack([o[0] for o in out])
            target_policies = torch.tensor([o[1][:-1] for o in out], dtype=torch.float)
            target_values = torch.tensor([[o[1][-1]] for o in out], dtype=torch.float)
            return features, target_policies, target_values

        if self.conv_input:
            features = torch.stack([o[0] for o in out])
            target_policies = torch.tensor([o[1] for o in out], dtype=torch.float)
            return features, target_policies

        if self.include_value:
            features = torch.tensor([o[0] for o in out], dtype=torch.float)
            target_policies = torch.tensor([o[1][:-1] for o in out], dtype=torch.float)
            target_values = torch.tensor([[o[1][-1]] for o in out], dtype=torch.float)
            return features, target_policies, target_values

        return torch.tensor([o[0] for o in out], dtype=torch.float), torch.tensor([o[1] for o in out], dtype=torch.float)

    def save(self, s, D):
        self.training_cases.append([s, D])
        if len(self.training_cases) > TrainingConf.buffer_max_size:
            self.training_cases.pop(0)

    def save_to_file(self):
        pickle.dump(self.training_cases, open(f"buffers/{TrainingConf.file_prefix}", "wb"))

if __name__ == '__main__':
    test = torch.ones(5, 5, 5)
    test2 = torch.zeros(5, 5, 5)
    test3 = torch.stack((test, test2))
    print(test.size())
    print(test2.size())
    print(test3.size())