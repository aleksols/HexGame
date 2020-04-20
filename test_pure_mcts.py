from tqdm import tqdm
from board import Board
from mcts_kernel import MCTS
import random
from node import Node

class DummyAnet():
    def predict(self, x):
        return [random.random() for _ in range(16)]


def play():
    wins = 0
    for game in tqdm(range(1, 100 + 1)):
        player = 1
        actual_board = Board(4, 1)
        board = Board(4, 1)
        kernel = MCTS(board, 1, DummyAnet())
        root = Node(board.state, None, board.player)

        while not actual_board.finished:
            board.set_state(actual_board.state)
            kernel.search(500, root)
            root = max(root.children, key=lambda x: x.visits)
            actual_board.play(root.edge_action)
            root.parent = None
            root.edge_action = None
        if board.next_player == 1:
            wins += 1
    print("Player 1 wins", wins, "of", game)


if __name__ == '__main__':
    play()