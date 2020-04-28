import torch.nn as nn
import torch

# TODO add pivotals for easy access

class GameConf:
    size = 5
    starting_player = 1


class MctsConf:
    simulations = 500
    actual_games = 600
    c = 1
    predict_value_eps = 0.5


class TrainingConf:
    num_anets = 5
    batch_size = 128
    buffer_max_size = 4000
    epochs = 1
    visualize = []
    verbose = 1
    save_directory = f"demo"
    file_prefix = f"demo_{GameConf.size}"


class TournamentConf:
    games = 100
    agent_policies = "probabilistic"
    directory = TrainingConf.save_directory
    file_prefix = TrainingConf.file_prefix
    visualize = []  # Index of games to be visualized


class NetworkConf:
    hidden_dims = [256, 256]
    activations = ["relu"] * 2
    output_activation = "softmax"
    output_args = {"dim": 1}
    optimizer = "adam"
    loss = "crossentropy"
    lr = 0.008
    network_type = "linear"  # linear, conv or residual
    predict_value = False


class ConvNetConf(NetworkConf):
    channel_conf = [128, 128, 128, 128, 1]
    activations = ["relu"] * 5
    kernel_sizes = [5, 3, 3, 3, 1]
    paddings = [2, 1, 1, 1, 0]


class ResNetConf:
    input_conf = ([(5, 256)], nn.ReLU())
    input_activation = nn.ReLU()
    res_conf = ([(256, 256), (256, 256)], nn.ReLU())
    num_res_blocks = 1
    policy_conf = (256, 2, nn.ReLU())
    value_conf = (256, 1, nn.ReLU())
    optimizer = torch.optim.Adam
    lr = 0.005
    c = 1e-4
