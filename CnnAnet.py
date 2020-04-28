from config import ConvNetConf, GameConf
import torch
import torch.nn as nn
from anet import ANET


class ConvAnet(ANET):
    def __init__(self,
                 channel_conf=ConvNetConf.channel_conf,
                 activations=ConvNetConf.activations,
                 kernel_sizes=ConvNetConf.kernel_sizes,
                 paddings=ConvNetConf.paddings
                 ):
        self.kernels = kernel_sizes
        self.paddings = paddings
        super().__init__(h_dims=channel_conf, act_func=activations)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=ConvNetConf.lr)
        self.loss_function = self.losses[ConvNetConf.loss]

    def _init_network(self):
        model = nn.Sequential()
        model.add_module("L0", nn.Conv2d(5, self.h_dims[0], padding=self.paddings[0], kernel_size=self.kernels[0]))
        for i in range(1, len(self.h_dims)):
            layer_name = f"L{i}"
            act_name = f"L{i - 1}_L{i}"
            in_channels = self.h_dims[i - 1]
            out_channels = self.h_dims[i]
            padding = self.paddings[i]
            kernel_size = self.kernels[i]
            model.add_module(act_name, self.activations[self.act_func[0]]())
            model.add_module(layer_name, nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size))
        return model

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, GameConf.size ** 2)
        return nn.Softmax(dim=0)(x)

    def predict(self, x):
        x = x.unsqueeze(dim=0)
        x = self.forward(x)
        return x[0]

    def _forward_without_softmax(self, x):
        x = self.model(x)
        x = x.view(-1, GameConf.size ** 2)
        return x


if __name__ == '__main__':
    dummy1 = torch.ones(2, 74)
    dummy2 = torch.ones(12800, 5, 6, 6)
    net1 = ANET()
    net2 = ConvAnet()
    print(net2.model[0])
