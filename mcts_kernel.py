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
        # if self.board.state == [1, 1, 0, 0, 0, 0, 0, 0, 0, 2]:
        #     self.anet.print_weigths()

        prediction = self.anet.predict(self.board.nn_state)
        valid_actions = self.board.valid_actions
        # dist = [0 for _ in range(len(prediction))]
        dist = torch.zeros(len(prediction), dtype=torch.float)
        for i in valid_actions:
            dist[i] = prediction[i]
        dist = normalize(dist, dim=0, p=1)
        # print("dist", dist)
        # print("valid", valid_actions)
        if dist.sum() == 0:
            return random.choice(valid_actions)
        action = random.choices(range(len(prediction)), weights=dist.tolist())[0]
        # if action not in valid_actions:
        #     print(self.board.nn_state)
        #     print(prediction.tolist())
        #     print(dist)
        #     print(valid_actions)
        #     self.anet.save("error")
        return action


    def backprop(self, leaf_node: Node, z):
        while leaf_node is not None:
            leaf_node.visits += 1
            leaf_node.wins += z
            leaf_node = leaf_node.parent

if __name__ == '__main__':
    # from anet import ANET
    # from board import Board
    # net = ANET.load("models/3_anet_error")
    # print(net)
    # b = Board(3, 1)
    # b.set_state([1, 1, 0, 0, 0, 0, 0, 0, 0, 2])
    # print(net.predict(b.nn_state))
    # inn = torch.tensor([b.nn_state], dtype=torch.float)
    # print("inn", inn)
    # import pprint
    # for i, m in enumerate(net.model.modules()):
    #     if i == 0:
    #         continue
    #     print("m", type(m))
    #     if i % 2:
    #         print("weights", m.weight)
    #     inn = m.forward(inn)
    #     print("result", inn)
    # pprint.pprint(dir(net.model))
    population = [0.01, 0.2, 0.4, 0.2, 0.1, 0.09]
    population = [i * 2 for i in population]
    print(sum(population))
    chosen = [0, 0, 0, 0, 0, 0]
    num = 1000000
    for i in range(num):
        c = random.choices(range(len(population)), weights=population)[0]
        chosen[c] += 1
    print(chosen)
    print([c / num for c in chosen])