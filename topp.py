import tensorflow as tf
import torch

from agent import Agent
from anet import ANET_v2
from board import Board


def get_model_path(filetype):
    from os import listdir
    return ["models/" + f for f in filter(lambda x: x[-len(filetype):] == filetype, listdir("models/"))]

def get_agents(anet_type):

    agents = {}
    model_paths = get_model_path("dat")
    for path in model_paths:
        agents[path] = Agent(TorchImplementation.load(path))

    return agents


def play_single(agent1, agent2, starting_player=1):
    board = Board(4, starting_player)
    action_sequence = []
    while not board.finished:
        state = board.state
        if board.player == 1:
            action = agent1.best_action(board.valid_actions, [state])
        else:
            action = agent2.best_action(board.valid_actions, [state])
        action_sequence.append(action)
        board.play(action)
    for i in range(1, 17, 4):
        print(board.state[i: i + 4])
    if board.next_player == 1:
        return 1, 0, action_sequence  # agent1 points, agent2 points and action sequence
    return 0, 1, action_sequence

def tournament():
    pass

if __name__ == '__main__':
    agents = get_agents("tf")
    b = Board(4, 1)
    state = b.state
    print(b.state)
    for key, value in get_agents("tf").items():
        print("Model from", key, ":", value.best_action(b.valid_actions, [state]))
    a1 = agents["models/anet_0.h5"]
    a2 = agents["models/anet_200.h5"]
    # a1 = Agent(m1)
    # a2 = Agent(m2)
    p1, p2, seq = play_single(a1, a2)
    print(p1, p2)
    for action in seq:
        print(action)

