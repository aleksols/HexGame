import math
import numpy as np
import torch
from torch.nn.functional import normalize

from board import Board
from node import Node
import random
# from sklearn.preprocessing import normalize


class MCTS:
    def __init__(self, board: Board, c, anet):
        self.board = board
        self.c = c
        self.anet = anet


    def search(self, num_simulations, root):
        # print(num_simulations, root)
        for _ in range(num_simulations):
            leaf_node = self.tree_search(root)
            z = self.rollout(leaf_node)
            self.backprop(leaf_node, z)
        # self.board.set_state(root.state)
        # return self.select_node(root, c=0)
        # collected = gc.collect()
        # print(collected)
        # print(gc.get_count())

    def select_node(self, node: Node, c) -> Node:  # Tree policy
        if node.player == 1:
            best_child = np.argmax([child.value + c * (math.log(node.N()) / child.N()) ** (1/2) for child in node.children])
        else:
            best_child = np.argmin([child.value - c * (math.log(node.N()) / child.N()) ** (1/2) for child in node.children])
        return node.children[best_child]

    def tree_search(self, root: Node):
        self.board.set_state(root.state)
        while not self.board.finished:
            if not root.expanded:
                new_node = root.expand(self.board.generate_child_states())
                return new_node
            root = self.select_node(root, self.c)
            self.board.set_state(root.state)
        return root

    def rollout(self, current_node: Node):
        self.board.set_state(current_node.state)
        while not self.board.finished:
            action = self.default_policy()
            self.board.play(action)
        if self.board.next_player == 2:
            return -1
        return 1

    def default_policy(self):
        prediction = self.anet.predict(self.board.nn_state)
        valid_actions = self.board.valid_actions
        dist = torch.zeros(len(prediction), dtype=torch.float)
        for i in valid_actions:
            dist[i] = prediction[i]
        dist = normalize(dist, dim=0, p=1)
        if dist.sum() == 0:
            return random.choice(valid_actions)
        action = random.choices(range(len(prediction)), weights=dist.tolist())[0]

        return action


    def backprop(self, leaf_node: Node, z):
        while leaf_node is not None:
            leaf_node.visits += 1
            leaf_node.wins += z
            leaf_node = leaf_node.parent

    def update_eps(self):  # Just so that this method can be called in training loop without problems
        pass
