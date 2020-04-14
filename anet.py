# import tensorflow as tf
from config import network
import config
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

class ANET_v2(nn.Module):
    optimizers = {
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop
    }

    activations = {
        "relu": nn.ReLU,
        "softmax": nn.Softmax
    }

    def __init__(self):
        super().__init__()
        self.model = self._init_network()
        self.optimizer = self.optimizers[network["optimizer"]](self.model.parameters())
        self.loss_function = nn.CrossEntropyLoss()  # TODO make generic

    def _init_network(self):
        modules = nn.ModuleList()
        dims = network["dimensions"]
        act = network["activations"]
        modules.append(nn.Linear(dims[0], dims[1]))
        modules.append(self.activations[act[0]]())
        for i in range(1, len(dims) - 2):
            modules.append(nn.Linear(dims[i], dims[i+1]))
            modules.append(self.activations[act[i]]())
        modules.append(nn.Linear(dims[-2], dims[-1]))
        modules.append(self.activations[act[-1]](dim=1))
        return nn.Sequential(*modules)

    def forward(self, x):
        x = torch.tensor([x], dtype=torch.float)
        # print(x.shape)
        return self.model(x)

    def train_model(self, batch, epochs):
        for i in range(epochs):
            for x, y in batch:
                self.optimizer.zero_grad()
                # print(x)
                # print(y)
                y = torch.argmax(torch.tensor(y)).view(1)
                # print("Fixed", y)
                out = self(x)
                # print("out ", out)
                # out = out.view(1, list(out.size())[0])
                # print("fixed x", out)
                loss = self.loss_function(out, y)
                # print("loss", loss)
                loss.backward()
                self.optimizer.step()


    def save(self, i):
        with open(f"models/anet_{i}", "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))

class ANET:
    def __init__(self):
        raise NotImplementedError

    def predict(self, x, **kwargs):
        raise NotImplementedError

    def fit(self, x, y, **kwargs):
        raise NotImplementedError

    def save(self, filepath, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load(filepath, **kwargs):
        raise NotImplementedError


# class TFImplementation(tf.keras.models.Sequential, ANET):
#
#     @staticmethod
#     def load(filepath, **kwargs):
#         return tf.keras.models.load_model(filepath)
#
#     def __init__(self):
#         super().__init__()
#         self.add(tf.keras.layers.Input(shape=(config.general["size"] ** 2 + 1)))
#         for layer, dim, activation in zip(network["layers"], network["dimensions"], network["activations"]):
#             self.add(layer(dim, activation=activation))
#         self.compile(optimizer=network["optimizer"], loss=network["loss"])
#
#     def predict(self, x, **kwargs):
#         # TODO the predict method in tensorflow is super slow. Try to do matrix multiplication manually
#         start = time.time()
#         print("predict", x)
#         predicted = super().predict([x], batch_size=1)[0]
#         prediction_time = time.time() - start
#         print("TF prediction time", prediction_time)
#         return predicted


class TorchImplementation(nn.Module, ANET):
    optimizers = {
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop
    }

    activations = {
        "relu": nn.ReLU,
        "softmax": nn.Softmax
    }

    def __init__(self):
        super().__init__()
        self.model = self._init_network()

    def _init_network(self):
        modules = nn.ModuleList()
        layers = network["layers"]
        dims = network["dimensions"]
        act = network["activations"]
        modules.append(nn.Linear(17, 64))
        modules.append(self.activations["relu"]())
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(dims[i], dims[i+1]))
            modules.append(self.activations[act[i]]())
        return nn.Sequential(*modules)

    def predict(self, x, **kwargs):
        start = time.time()
        predicted = self.model.forward(torch.tensor([x], dtype=torch.float))
        pred_time = time.time() - start
        print("Pytorch prediction time", pred_time)
        return predicted

    def fit(self, x, y, **kwargs):
        # TODO implement
        pass

    def save(self):
        # TODO implement
        pass

    @staticmethod
    def load(self):
        # TODO implement
        pass

if __name__ == '__main__':
    from board import Board
    b = Board(config.general["size"], 1)
    state = b.state
    anet = ANET_v2()
    print(anet)
    import random
    x = []
    y = []
    for i in range(1):
        x.append([random.randint(0, 2) for _ in range(17)])
        y_ = [0]*16
        y_[random.randint(0, 15)] = 1
        y.append(y_)
    batch = zip(x, y)
    # batch = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    # print(batch[0].shape)
    # print(batch[1].shape)
    # print(torch.randn(10))
    # print(torch.randn(10).view(1, -1))
    # print(net(batch[0]).shape)
    anet.train_model(batch, 3)
    print(anet.forward(x[0]).sum())
    # print(type(net))
    # for i in range(100):
    #     print(net.forward(state))


