import tensorflow as tf
from config import network
import config
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class TFImplementation(tf.keras.models.Sequential, ANET):

    @staticmethod
    def load(filepath, **kwargs):
        return tf.keras.models.load_model(filepath)

    def __init__(self):
        super().__init__()
        self.add(tf.keras.layers.Input(shape=(config.general["size"] ** 2 + 1)))
        for layer, dim, activation in zip(network["layers"], network["dimensions"], network["activations"]):
            self.add(layer(dim, activation=activation))
        self.compile(optimizer=network["optimizer"], loss=network["loss"])

    def predict(self, x, **kwargs):
        # TODO the predict method in tensorflow is super slow. Try to do matrix multiplication manually
        start = time.time()
        print("predict", x)
        predicted = super().predict([x], batch_size=1)[0]
        prediction_time = time.time() - start
        print("TF prediction time", prediction_time)
        return predicted


class TorchImplementation(nn.Module, ANET):
    optimizers = {
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        # "rmsprop": torch.optim.RMSProp
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
    tf = TFImplementation()
    pytorch = TorchImplementation()
    print(pytorch.model.parameters())
    for i in range(100):
        print("tf", tf.predict(state))
        print("pytorch", pytorch.predict(state))

