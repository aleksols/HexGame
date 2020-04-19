from config import network
import config
import torch
import torch.nn as nn
import pickle
import time


class PaddedConv(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size):
        super().__init__(in_channels, out_channels, kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


class PaddedNormConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        self.add_module(
            "conv", nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)
        )
        self.add_module("batch_norm", nn.BatchNorm2d(out_channels))


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
        "tanh": nn.Tanh,
        "linear": nn.Linear
    }

    def __init__(self):
        super().__init__()
        self.model = self._init_network()
        self.optimizer = self.optimizers[network["optimizer"]](self.model.parameters(), lr=network["learning rate"])
        self.loss_function = nn.CrossEntropyLoss()  # TODO make generic

    def _init_network(self):
        modules = nn.ModuleList()
        dims = network["dimensions"]
        act = network["activations"]
        for i in range(len(dims) - 2):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(self.activations[act[i]]())
        modules.append(nn.Linear(dims[-2], dims[-1]))
        modules.append(self.activations[act[-1]](**config.network["output args"]))
        return nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):  # Use predict for lists and forward for tensors
        x = torch.tensor([x], dtype=torch.float)
        return self.model(x)[0]

    def train_model(self, x, y, epochs, verbose=0):
        start = time.time()
        for i in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.custom_cross_entropy(out, y)

            if verbose == 2:
                print("loss", loss, i)
            loss.backward()
            self.optimizer.step()
        if verbose == 1:
            print("Trained on",
                  len(x),
                  "cases for",
                  epochs,
                  "epochs",
                  time.time() - start,
                  "seconds"
                  )

    def custom_cross_entropy(self, pred, target):
        return -(target * torch.log(pred)).sum(dim=1).mean()

    def save(self, i):
        with open(f"models/{config.training['file prefix']}_anet_{i}", "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))


class ConvAnet(ANET):
    def _init_network(self):
        model = nn.Sequential()
        dims = network["dimensions"]
        act = network["activations"]
        model.add_module("input", PaddedNormConv(in_channels=2, out_channels=dims[0]))
        model.add_module("input_activation", self.activations[act[0]]())
        for i in range(len(dims) - 2):
            model.add_module(f"hidden_{i}", PaddedNormConv(in_channels=dims[i], out_channels=dims[i + 1]))
            model.add_module(f"hidd_act_{i}", self.activations[act[i]]())
        return model


if __name__ == '__main__':
    from board import Board
    b = Board(3, 1)
    b.play(2)
    b.play(1)
    b.play(0)
    b.play(7)
    bnet = ANET()
    pred = bnet.predict(b.nn_state)
    print(pred)
    anet = ConvAnet()
    print(anet)
    dummy = torch.zeros((1, 2, 3, 3))
    print("test", dummy[0, 0])
    dummy[0, 0, 1, 1] = 1
    dummy[0, 0, 0, 1] = 2
    dummy[0, 1] = 1.
    # dummy[]
    print(dummy)
    print("Autopadded", anet(dummy))
