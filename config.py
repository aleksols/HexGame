import torch.nn as nn
import torch


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


# TODO implement agent strats


# tournament = {
#     "games": 100,
#     "agent policies": "probabilistic",  # greedy, probabilistic or random
#     "file prefix": f"{game['size']}",
#     "visualize": []
# }

# network = {
#     "h_dims": [128, 128],
#     "activations": ["relu"] * 2,
#     "output func": "softmax",
#     "output args": {"dim": 1},
#     "optim": "adam",
#     "loss": "categorical_crossentropy",
#     "learning rate": 0.008,
#     "use conv": True,
#     "predict value": False
# }

# training = {
#     "batch size": 128,
#     "epochs": 1,
#     "simulations": 50,
#     "actual games": 60,
#     "c": 1,
#     "num anets": 5,
#     "file prefix": f"test_refactor{game['size']}",
#     "visualize": [],  # What games to visualize. F.eks [1, 50, 200]. 1 indexed
#     "buffer max size": 4000,
#     "eps": 0.0,
#     "eps decay": 0.9
# }

# conv_net = {
#     "h_channels": [64],
#     "activations": ["relu"],
#     "output func": "softmax",
#     "output args": {"dim": 1},
#     "optim": "adam",
#     "loss": "custom_crossentropy",
#     "learning rate": 0.0001,
# }


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

# worked pretty well for greedy agents and size 3
# network = {
#     "h_dims": [64, 64],
#     "activations": ["relu"] * 2,
#     "output func": "softmax",
#     "output args": {"dim": 1},
#     "optim": "adam",
#     "loss": "categorical_crossentropy",
#     "learning rate": 0.001,
#     "use conv": False
# }
#
# training = {
#     "batch size": 32,
#     "epochs": 1,
#     "simulations": 500,
#     "actual games": 200,
#     "num anets": 5,
#     "file prefix": f"test_adam{game['size']}",
#     "visualize": [],  # What games to visualize. F.eks [1, 50, 200]. 1 indexed
#     "buffer max size": 2000
# }
