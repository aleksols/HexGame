import torch
import torch.nn as nn
from config import ResNetConf, TrainingConf, GameConf
# from anet import custom_cross_entropy
import time
# from anet import ANET
import pickle

class BasicBlock(nn.Sequential):
    def __init__(self, channel_conf: iter, activation=nn.ReLU()):
        super().__init__()
        for i, (in_channels, out_channels) in enumerate(channel_conf[:-1]):
            self.add_module(f"conv{i}", nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3, bias=False))
            self.add_module(f"bn{i}", nn.BatchNorm2d(out_channels))
            self.add_module(f"act{i}", activation)
        in_channels, out_channels = channel_conf[-1]
        self.add_module(f"conv{len(channel_conf) - 1}",
                        nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3, bias=False))
        self.add_module(f"bn{len(channel_conf) - 1}", nn.BatchNorm2d(out_channels))

    def squared_weight_sum(self):
        weight_sum = torch.zeros(1)
        for m in self.modules():
            if "weight" in dir(m):
                weight_sum += (m.weight ** 2).sum()
        return weight_sum


class ResBlock(nn.Module):
    def __init__(self, block_conf: iter, activation=nn.ReLU(), block_activation=nn.ReLU()):
        super().__init__()
        self.block = BasicBlock(block_conf, block_activation)
        self.activation = activation

    def forward(self, x):
        residual = x
        block_result = self.block(x)
        out = residual + block_result
        return self.activation(out)

    def squared_weight_sum(self):
        return self.block.squared_weight_sum() + 1


class PolicyBlock(nn.Sequential):
    def __init__(self, in_channels, filters, activation=nn.ReLU()):
        super().__init__()
        self.conv_filter = nn.Conv2d(in_channels, filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(filters)
        self.activation = activation
        self.policy_out = nn.Linear(filters * GameConf.size ** 2, GameConf.size ** 2)
        self.filters = filters

    def forward(self, x):
        x = self.conv_filter(x)
        x = self.bn(x)
        x = self.activation(x)
        x = x.view(-1, self.filters * GameConf.size ** 2)
        x = self.policy_out(x)
        return x

    def squared_weight_sum(self):
        weight_sum = torch.zeros(1)
        weight_sum += (self.conv_filter.weight ** 2).sum()
        weight_sum += (self.bn.weight ** 2).sum()
        weight_sum += (self.policy_out.weight ** 2).sum()
        return weight_sum


class ValueBlock(nn.Sequential):
    def __init__(self, in_channels, filters, activation=nn.ReLU()):
        super().__init__()
        self.conv_filter = nn.Conv2d(in_channels, filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(filters)
        self.activation = activation
        self.dense_layer = nn.Linear(filters * GameConf.size ** 2, filters * GameConf.size ** 2)
        self.value_out = nn.Linear(filters * GameConf.size ** 2, 1)
        self.filters = filters

    def forward(self, x):
        x = self.conv_filter(x)
        x = self.bn(x)
        x = self.activation(x)
        x = x.view(-1, self.filters * GameConf.size ** 2)
        x = self.dense_layer(x)
        x = self.activation(x)
        x = self.value_out(x)
        return torch.tanh(x)

    def squared_weight_sum(self):
        weight_sum = torch.zeros(1)
        weight_sum += (self.conv_filter.weight ** 2).sum()
        weight_sum += (self.bn.weight ** 2).sum()
        weight_sum += (self.dense_layer.weight ** 2).sum()
        weight_sum += (self.value_out.weight ** 2).sum()
        return weight_sum


class ResNet(nn.Module):
    def __init__(self, input_conf: tuple, input_activation, res_conf: tuple, num_blocks: int, policy_conf: tuple,
                 value_conf: tuple):
        super().__init__()
        self.input_block = BasicBlock(*input_conf)
        self.input_activation = input_activation
        self.res_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.res_blocks.append(ResBlock(*res_conf))
        self.policy_block = PolicyBlock(*policy_conf)
        self.value_block = ValueBlock(*value_conf)
        self.optimizer = ResNetConf.optimizer(self.parameters(), lr=ResNetConf.lr)

    def forward(self, x):
        x = self.base(x)
        value = self.value_block(x)
        policy = self.policy_block(x)
        return value, nn.Softmax(dim=1)(policy)

    def base(self, x):
        x = self.input_block(x)
        x = self.input_activation(x)
        for block in self.res_blocks:
            x = block(x)
        return x

    def policy(self, x):
        x = self.base(x)
        return self.policy_block(x)

    def predict(self, x):
        x = x.unsqueeze(dim=0)
        x = self.policy(x)
        return nn.Softmax(dim=1)(x)[0]

    def value(self, x):
        x = x.unsqueeze(dim=0)
        x = self.base(x)
        return self.value_block(x)

    def squared_weight_sum(self):
        weight_sum = torch.zeros(1)
        weight_sum += self.input_block.squared_weight_sum()
        for block in self.res_blocks:
            weight_sum += block.squared_weight_sum()
        weight_sum += self.policy_block.squared_weight_sum()
        weight_sum += self.value_block.squared_weight_sum()
        return weight_sum.squeeze()

    def cross_entropy(self, pred, target):
        return - (pred * torch.log(target)).sum(dim=1)

    def resnet_loss(self, pred_policy, pred_value, target_policy, target_value):
        value_loss = nn.functional.mse_loss(pred_value, target_value)
        target_policy = target_policy.argmax(dim=1)
        policy_loss = nn.functional.cross_entropy(pred_policy, target_policy)
        # regularization is part of optimizer and is therefore not here
        return value_loss + policy_loss

    def train_model(self, features, target_policies, target_values, epochs, verbose=0):
        # print(features.size())
        # print(target_policies.size())
        # print(target_values.size())
        start = time.time()
        loss = 0
        for i in range(epochs):
            self.optimizer.zero_grad()

            value, policy = self.forward(features)
            loss = self.resnet_loss(policy, value, target_policies, target_values)

            if verbose == 2:
                print("loss", loss, i)

            loss.backward()
            self.optimizer.step()
        if verbose == 1:
            print("Trained on",
                  len(features),
                  "cases for",
                  epochs,
                  "epochs in",
                  time.time() - start,
                  "seconds. Last loss:",
                  loss
                  )
        return loss

    def save(self, i):
        with open(f"models/{TrainingConf.file_prefix}_anet_{i}", "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))

if __name__ == '__main__':
    from config import ResNetConf
    input_conf = ([(18, 256)], nn.ReLU())
    input_activation = nn.ReLU()
    res_conf = ([(256, 256), (256, 256)], nn.ReLU())
    num_res_blocks = 7
    policy_conf = (256, 2, nn.ReLU())
    value_conf = (256, 1, nn.ReLU())
    net = ResNet(
        ResNetConf.input_conf,
        ResNetConf.input_activation,
        ResNetConf.res_conf,
        ResNetConf.num_res_blocks,
        ResNetConf.policy_conf,
        ResNetConf.value_conf)
    print(net)
    # print(len([i for i in net.parameters(recurse=True)]))
    # dummy = torch.ones(10, 18, GameConf.size, GameConf.size)
    # dummy_values = torch.ones((10, 1))
    # dummy_policies = torch.rand((10, GameConf.size ** 2))
    # print(dummy_policies.argmax(dim=1))
    # print(net(dummy))
    # net.train_model(dummy, dummy_policies, dummy_values, 1, verbose=2)
    # print(net(dummy))

    dummy = torch.rand(10, 25)
    lin = nn.Linear(25, 2)
    target = torch.ones(10, 2)

    out = lin(dummy)
    print(net.cross_entropy(out, target))
    target = target.argmax(dim=1)
    print(dummy.size())
    print(target.size())
    print(out.size())
    print(target)
    loss = nn.functional.cross_entropy(out, target)
    print(loss)



