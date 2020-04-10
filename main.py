import config
from board import Board
from mcts_kernel import MCTS
from node import Node
from replay_buffer import ReplayBuffer
from anet import ANET
from state_manager import StateManager
from tqdm import tqdm
import numpy as np
import concurrent.futures


def train():
    i = config.general["actual games"] // (config.general["num anets"] - 1)
    replay_buffer = ReplayBuffer()
    anet = ANET()
    anet.save("models/anet_0.h5")
    anet.summary()

    n_threads = 8
    sims_per_thread = config.general["simulations"] // n_threads
    rest = config.general["simulations"] % n_threads
    num_sims = [sims_per_thread] * (n_threads - 1) + [sims_per_thread + rest]

    actual_board = Board(config.general["size"], starting_player=1)
    mc_board = Board(config.general["size"], starting_player=1)

    for game in tqdm(range(1, config.general["actual games"] + 1)):
        actual_board.reset()
        mc_board.reset()
        mct = MCTS(mc_board, c=1, anet=anet)
        root = Node(mc_board.state, None, mc_board.player == 1)

        while not actual_board.finished:
            mc_board.set_state(actual_board.state)
            mct.search(config.general["simulations"], root)

            D = [0 for _ in range(actual_board.size ** 2)]
            for child in root.children:
                D[child.edge_action] = child.visits / root.visits
            replay_buffer.save(root.state, D)
            root = max(root.children, key=lambda x: x.visits / root.visits)  # Or some other algorithm for choosing action
            actual_board.play(root.edge_action)  # Updating B_a to s*

            # Discard the rest of the tree and make s* the root
            root.parent = None
            root.edge_action = None

        x, y = replay_buffer.get_random_minibatch(config.network["batch size"])
        anet.fit(x, y, verbose=1)
        if game % i == 0:
            anet.save(f"models/anet_{game}.h5")

if __name__ == '__main__':
    train()