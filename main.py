from config import ResNetConf, NetworkConf, MctsConf, TrainingConf, GameConf
from board import Board, ResBoard
from mcts_kernel import MCTS
from mcts_kernel_resnet import MCTSResNet
from node import Node
from replay_buffer import ReplayBuffer
from anet import ANET
from residual_anet import ResNet
from tqdm import tqdm
from visualize import visualize
import torch.nn.functional as F
import torch
from CnnAnet import ConvAnet

def get_save_interval():
    i = MctsConf.actual_games // (TrainingConf.num_anets - 1)
    if i == 0:
        i = 1
    return i


def init_replay_buffer():
    conv_input = NetworkConf.network_type == "conv" or NetworkConf.network_type == "residual"
    return ReplayBuffer(NetworkConf.predict_value, conv_input)


def init_actor_network():
    if NetworkConf.network_type == "conv":
        return ConvAnet()
    if NetworkConf.network_type == "residual":
        return ResNet(
            ResNetConf.input_conf,
            ResNetConf.input_activation,
            ResNetConf.res_conf,
            ResNetConf.num_res_blocks,
            ResNetConf.policy_conf,
            ResNetConf.value_conf
        )
    return ANET()


def init_boards(starting_player):
    if NetworkConf.network_type == "conv" or NetworkConf.network_type == "residual":
        actual = ResBoard(GameConf.size, starting_player)
        mc = ResBoard(GameConf.size, starting_player)
        return actual, mc
    actual = Board(GameConf.size, starting_player)
    mc = Board(GameConf.size, starting_player)
    return actual, mc


def init_monte_carlo(board, anet):
    if NetworkConf.predict_value:
        return MCTSResNet(board, anet, c=MctsConf.c, eps=MctsConf.predict_value_eps)
    return MCTS(board, c=1, anet=anet)


def get_distribution(root):
    D = [0 for _ in range(GameConf.size ** 2)]
    for child in root.children:
        D[child.edge_action] = child.visits
    if NetworkConf.predict_value:
        return F.normalize(torch.tensor(D, dtype=torch.float), dim=0, p=1).tolist() + [root.value]
    return F.normalize(torch.tensor(D, dtype=torch.float), dim=0, p=1).tolist()


def train():
    prompt = "You are about to start a training session. Are you sure you want this? (yes/no) "
    if input(prompt) != "yes":
        return

    i = get_save_interval()
    replay_buffer = init_replay_buffer()
    anet = init_actor_network()
    print(anet)
    anet.save(0)

    losses = []

    for game in tqdm(range(1, MctsConf.actual_games + 1)):
        actual_board, mc_board = init_boards((game - GameConf.starting_player) % 2 + 1)

        mct = init_monte_carlo(mc_board, anet)
        root = Node(mc_board.state, None, mc_board.player == 1)
        action_sequence = []

        while not actual_board.finished:
            mc_board.set_state(actual_board.state)
            mct.search(MctsConf.simulations, root)

            D = get_distribution(root)

            replay_buffer.save(actual_board.nn_state, D)
            root = mct.select_node(root, c=0)  # Or some other algorithm for choosing action
            actual_board.play(root.edge_action)  # Updating B_a to s*
            action_sequence.append(root.edge_action)

            # Discard the rest of the tree and make s* the root
            root.parent = None
            root.edge_action = None

        if game in TrainingConf.visualize:
            visualize(actual_board, action_sequence, f"Game {game}")

        training_data = replay_buffer.get_random_minibatch(TrainingConf.batch_size)
        loss = anet.train_model(*training_data, epochs=TrainingConf.epochs, verbose=TrainingConf.verbose)
        losses.append(loss)
        mct.update_eps()
        if game % i == 0:
            anet.save(game)

    replay_buffer.save_to_file()  # For debugging purposes later
    import matplotlib.pyplot as plt
    plt.plot(losses)
    avg = [losses[0]]
    for i in range(1, len(losses)):
        avg.append((sum(avg[-9:]) + losses[i]) / (min(10, len(avg) + 1)))
    plt.plot(avg)
    plt.show()
    plt.plot(losses)
    plt.plot(avg)
    plt.savefig(f"models/{TrainingConf.save_directory}/loss.png")


if __name__ == '__main__':
    train()
