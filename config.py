import torch.nn as nn
import torch

# Pivotals for easy access
k = 5
mcts_sims = 10
mcts_episodes = 15
mcts_exploration = 1
learn_rate = 0.008
layer_conf = [128, 128]
activation_functions = ["relu", "linear"]
network_optimizer = "rmsprop"
number_of_anets = 4
tournament_games = 50
visualize_training = []
visualize_tournament = []  # ranges [1, tournament_games] for every series or [1001, ->} for game_ids
animation_speed = 100

class GameConf:
    size = k
    starting_player = 1


class MctsConf:
    simulations = mcts_sims
    actual_games = mcts_episodes
    c = mcts_exploration
    predict_value_eps = 0.5


class TrainingConf:
    num_anets = number_of_anets
    batch_size = 128
    buffer_max_size = 4000
    epochs = 1
    visualize = visualize_training
    verbose = 1
    save_directory = f"demo"
    file_prefix = f"demo_{GameConf.size}"


class TournamentConf:
    games = tournament_games
    agent_policies = "probabilistic"
    directory = TrainingConf.save_directory
    file_prefix = TrainingConf.file_prefix
    visualize = visualize_tournament  # Index of games to be visualized


class NetworkConf:
    hidden_dims = layer_conf
    activations = activation_functions
    output_activation = "softmax"
    output_args = {"dim": 1}
    optimizer = network_optimizer
    loss = "crossentropy"
    lr = learn_rate
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
