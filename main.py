import config
from board import Board
from mcts_kernel import MCTS
from node import Node
from replay_buffer import ReplayBuffer
from anet import ANET
from tqdm import tqdm
from visualize import visualize
import torch.nn.functional as F
import torch


def train():
    prompt = input("You are about to start a training session."
                   "Are you sure you want this? (yes/no) ")
    if prompt != "yes":
        return

    i = config.training["actual games"] // (config.training["num anets"] - 1)
    if i == 0:
        i = 1
    replay_buffer = ReplayBuffer()
    anet = ANET()
    anet.save(0)

    actual_board = Board(config.game["size"], starting_player=1)
    mc_board = Board(config.game["size"], starting_player=1)

    for game in tqdm(range(1, config.training["actual games"] + 1)):
        actual_board.reset()
        mc_board.reset()
        mct = MCTS(mc_board, c=1, anet=anet)
        root = Node(mc_board.state, None, mc_board.player == 1)
        action_sequence = []

        while not actual_board.finished:
            mc_board.set_state(actual_board.state)
            mct.search(config.training["simulations"], root)

            D = [0 for _ in range(actual_board.size ** 2)]
            # print("-------\n", root.visits)
            for child in root.children:
                D[child.edge_action] = child.visits
                # print(child.visits)
            # print("\n-------")
            D = F.normalize(torch.tensor(D, dtype=torch.float), dim=0, p=1)
            # print(D.tolist(), sum(D))
            replay_buffer.save(actual_board.nn_state, D.tolist())
            root = mct.select_node(root, c=0)  # Or some other algorithm for choosing action
            actual_board.play(root.edge_action)  # Updating B_a to s*
            action_sequence.append(root.edge_action)

            # Discard the rest of the tree and make s* the root
            root.parent = None
            root.edge_action = None

        if game in config.training["visualize"]:
            visualize(actual_board, action_sequence)

        x, y = replay_buffer.get_random_minibatch(config.training["batch size"])
        anet.train_model(x, y, config.training["epochs"])

        if game % i == 0:
            anet.save(game)
    replay_buffer.save_to_file()  # For debugging purposes later


if __name__ == '__main__':
    train()
