import config
from board import Board
from mcts_kernel import MCTS
from node import Node
from replay_buffer import ReplayBuffer
from anet import ANET
from state_manager import StateManager
from tqdm import tqdm
import numpy as np

def main():
    i = 5
    replay_buffer = ReplayBuffer()
    anet = ANET()
    actual_board = Board(config.general["size"], starting_player=1)
    mc_board = Board(config.general["size"], starting_player=1)

    for game in tqdm(range(config.general["actual games"])):
        actual_board.reset()


        state_manager = StateManager(mc_board)
        mct = MCTS(state_manager, c=1, anet=anet)
        root = Node(state_manager.game_state, None, state_manager.player)

        while not actual_board.final_state:
            mc_board.set_state(actual_board.state)
            state_manager.set_state(root.state, root.player)
            mct.search(config.general["simulations"], root)
            D = [child.visits / root.visits for child in root.children]
            replay_buffer.save(root.state, D)
            root = root.children[np.argmax(D)]  # Or some other algorithm for choosing action
            actual_board.play(root.edge_action)  # Updating B_a to s*

            # Discard the rest of the tree and make s* the root
            root.parent = None
            root.edge_action = None

        anet.fit(replay_buffer.get_random_minibatch())
        if game % i == 0:
            anet.save(f"models/anet_{game}.h5")
