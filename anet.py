from config import NetworkConf, GameConf, TrainingConf
import config
import torch
import torch.nn as nn
import pickle
import time
import math
from os import listdir, mkdir


def custom_cross_entropy(pred, target):
    pred = nn.functional.relu(pred) + 1e-10  # To prevent log(0)
    return -(target * torch.log(pred)).sum(dim=1).mean()


class ANET(nn.Module):
    #  TODO implement missing
    optimizers = {
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop
    }

    activations = {
        "relu": nn.ReLU,
        "softmax": nn.Softmax,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh
    }

    losses = {
        "crossentropy": nn.CrossEntropyLoss(),
        "custom_crossentropy": custom_cross_entropy,
        "kldiv": nn.KLDivLoss(reduction="batchmean")
    }

    def __init__(self, h_dims=NetworkConf.hidden_dims, act_func=NetworkConf.activations):
        super().__init__()
        self.h_dims = h_dims
        self.act_func = act_func
        self.model = self._init_network()
        self._init_weights()
        self.optimizer = self.optimizers[NetworkConf.optimizer](self.model.parameters(), lr=NetworkConf.lr)
        self.loss_function = self.losses[NetworkConf.loss]
        self.save_directory = TrainingConf.save_directory

    def _init_network(self):
        model = nn.Sequential()
        model.add_module("L0", nn.Linear(2 * GameConf.size ** 2 + 2, self.h_dims[0]))
        model.add_module("L0_L1", self.activations[self.act_func[0]]())
        for i in range(len(self.h_dims) - 2):
            layer_name = f"L{i + 1}"
            act_name = f"L{i}_L{i + 1}"
            model.add_module(layer_name, nn.Linear(self.h_dims[i], self.h_dims[i + 1]))
            model.add_module(act_name, self.activations[self.act_func[i + 1]]())
        layer_name = f"L{len(self.h_dims)}"
        act_name = f"output"
        model.add_module(layer_name, nn.Linear(self.h_dims[-1], GameConf.size ** 2))
        model.add_module(act_name, self.activations[NetworkConf.output_activation](**NetworkConf.output_args))
        return model

    def _init_weights(self):
        for m in self.model:
            if "weight" in dir(m):
                num_weights = m.weight.size()[1]
                with torch.no_grad():
                    m.weight *= math.sqrt(2 / num_weights)

    def forward(self, x):
        result = self.model(x)
        return result

    def _forward_without_softmax(self, x):
        for m in self.model[:-1]:
            x = m(x)
        return x

    def predict(self, x):  # Use predict for lists and forward for tensors
        x = torch.tensor([x], dtype=torch.float)
        return self.forward(x)[0]

    def transform_target(self, y):
        if NetworkConf.loss == "crossentropy":
            return y.argmax(dim=1)
        return y

    def train_model(self, x, y, epochs, verbose=0):
        start = time.time()
        loss = 0
        y = self.transform_target(y)  # Different loss functions takes inn different targets
        for i in range(epochs):
            self.optimizer.zero_grad()
            if NetworkConf.loss == "crossentropy":
                out = self._forward_without_softmax(x)
            elif NetworkConf.loss == "kldiv":
                out = nn.LogSoftmax(dim=1)(self._forward_without_softmax(x))
            else:
                out = self.forward(x)
            loss = self.loss_function(out, y)
            if verbose == 2:
                print("loss", loss, i)

            loss.backward()
            self.optimizer.step()
        if verbose >= 1:
            print("Trained on",
                  len(x),
                  "cases for",
                  epochs,
                  "epochs in",
                  time.time() - start,
                  "seconds. Last loss:",
                  loss
                  )
        return loss

    def save(self, i):
        if TrainingConf.save_directory not in listdir("models/"):
            mkdir("models/" + TrainingConf.save_directory)
        torch.save(self, f"models/{TrainingConf.save_directory}/{TrainingConf.file_prefix}_anet_{i}")

    @staticmethod
    def load(path):
        return torch.load(path)#open(path, 'rb'))
        # print(path)
        # return pickle.load(open(path, "rb"))

    def print_weigths(self):
        for m in self.model:
            if "weight" in dir(m):
                print("-------------------------------------------------------")
                print(m, m.weight.size())
                print(m.weight)
                print("-------------------------------------------------------\n\n\n\n\n")


# class PaddedConv(nn.Conv2d):
#     def __init__(self, in_channels: int, out_channels: int, kernel_size):
#         super().__init__(in_channels, out_channels, kernel_size)
#         self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
#
#
# class PaddedNormConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
#         super().__init__()
#         self.add_module(
#             "conv", nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)
#         )
#         self.add_module("batch_norm", nn.BatchNorm2d(out_channels))
#


# class ConvAnet(ANET):
#     def __init__(self, h_dims=conv_net["h_channels"], act_func=conv_net["activations"]):
#         super().__init__(h_dims, act_func)
#
#     def _init_network(self):
#         model = nn.Sequential()
#         dims = network["dimensions"]
#         act = network["activations"]
#         model.add_module("input", PaddedNormConv(in_channels=2, out_channels=dims[0]))
#         model.add_module("input_activation", self.activations[act[0]]())
#         for i in range(len(dims) - 2):
#             model.add_module(f"hidden_{i}", PaddedNormConv(in_channels=dims[i], out_channels=dims[i + 1]))
#             model.add_module(f"hidd_act_{i}", self.activations[act[i]]())
#         return model
#
#     def forward(self, x):
#         pass
#
#     def predict(self, x):  # input is board.nn_state
#         pass

if __name__ == '__main__':
    dummy = torch.ones(10, 2 * GameConf.size ** 2 + 2)
    net = ANET()
    from replay_buffer import ReplayBuffer
    import pickle
    rb = ReplayBuffer(False)
    rb.training_cases = pickle.load(open(f"buffers/{GameConf.size}", "rb"))
    x, y = rb.get_random_minibatch(5)
    net.train_model(x, y, 1, verbose=2)
    print(x.size())
    print(y.size())
    print(net)
    print(net(dummy))
    # conv = nn.Conv2d(256, 2, kernel_size=1)
    # bn = nn.BatchNorm2d(2)
    # print(conv.weight.size())
    # print(bn.weight)
    # print(bn.weight.size())
    # test1 = torch.ones(2, 4, 2, 2)
    # test2 = torch.ones(2, 4, 2, 2)
    # test = test1 + test2
    # print(test)
    # print(test ** 2)
    # print((test ** 2).sum())


    # value = ValueBlock(256, 1)
    # print(value)
    # print(value(dummy))
