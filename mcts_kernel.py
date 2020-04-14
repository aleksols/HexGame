import math
import numpy as np
from board import Board
from node import Node
from sklearn.preprocessing import normalize

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
        # self.state_manager.set_state(root.state, root.player)
        # return self.select_node(root, c=0)
        # collected = gc.collect()
        # print(collected)
        # print(gc.get_count())

    def select_node(self, node: Node, c) -> Node:  # Tree policy
        if node.player:
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
        prediction = self.anet.forward(self.board.state)[0]
        valid_actions = self.board.valid_actions
        dist = [0 for _ in range(len(prediction))]
        for i in valid_actions:
            dist[i] = prediction[i]
        dist = normalize([dist], norm="l1")[0]
        action = np.random.choice(range(len(prediction)), p=dist)
        # del prediction
        # del valid_actions
        # del dist
        return action


    def backprop(self, leaf_node: Node, z):
        while leaf_node is not None:
            leaf_node.visits += 1
            leaf_node.wins += z
            leaf_node = leaf_node.parent
