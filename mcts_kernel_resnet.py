import random
import config

from mcts_kernel import MCTS


class MCTSResNet(MCTS):
    def __init__(self, board, anet, c=1, eps=0.9):
        super().__init__(board, c, anet)
        self.eps = eps

    def search(self, num_simulations, root):
        for _ in range(num_simulations):
            leaf_node = self.tree_search(root)
            r = random.random()
            if r > self.eps:
                self.board.set_state(leaf_node.state)
                if self.board.finished:
                    z = 1 if self.board.next_player == 1 else -1
                else:
                    z = self.anet.value(self.board.nn_state).item()
            else:
                z = self.rollout(leaf_node)
            self.backprop(leaf_node, z)

    def update_eps(self):
        self.eps *= config.training["eps decay"]
