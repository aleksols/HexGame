import torch

import config
from agent import Agent
from anet import ANET
from board import Board
from visualize import visualize


def get_model_path():
    from os import listdir
    prefix = config.tournament["file prefix"]
    return ["models/" + f for f in filter(lambda x: x[:len(prefix)] == prefix, listdir("models/"))]


def get_agents(policy="probabilistic"):
    agents = {}
    model_paths = get_model_path()
    for path in model_paths:
        agents[path.replace("models/", "")] = Agent(ANET.load(path), policy)
    return agents


def play_single(agent1, agent2, starting_player=1, visual=False):
    board = Board(config.game["size"], starting_player)
    action_sequence = []
    while not board.finished:
        state = board.nn_state
        if board.player == 1:
            action = agent1.best_action(board.valid_actions, state, verbose=False)
        else:
            action = agent2.best_action(board.valid_actions, state, verbose=False)
        action_sequence.append(action)
        board.play(action)
    if visual:
        visualize(board, action_sequence)
    if board.next_player == 1:
        return 1, 0, action_sequence  # agent1 points, agent2 points and action sequence
    return 0, 1, action_sequence


def tournament():
    agents = get_agents(config.tournament["agent policies"])
    detailed = {}
    scores = {}
    for a in agents.keys():
        detailed[(a, "starting")] = 0
        detailed[(a, "not starting")] = 0
        scores[a] = 0
    # scores = {(a, "starting"): 0, (a,"not starting"): 0 for a in agents.keys()}

    num_games = config.tournament["games"]
    player_ones = list(agents.keys())
    player_twos = list(agents.keys())

    for i, agent1 in enumerate(player_ones[:-1], 1):
        for agent2 in player_twos[i:]:
            if agent1 == agent2:
                continue
            for game in range(1, num_games + 1):
                visual = game in config.tournament["visualize"]
                # print(agent1, "playing", agent2)
                # print(agents[agent1].anet)
                visual = agent1 == "test_adam3_anet_200" and game == 1
                a1_score, a2_score, seq = play_single(agents[agent1], agents[agent2], game % 2 + 1, visual)
                if game % 2 == 0:
                    detailed[(agent1, "starting")] += a1_score
                    detailed[(agent2, "not starting")] += a2_score
                else:
                    detailed[(agent1, "not starting")] += a1_score
                    detailed[(agent2, "starting")] += a2_score
                scores[agent1] += a1_score
                scores[agent2] += a2_score
    return scores, detailed

def play_manually():
    agent = get_agents("greedy")["test_adam3_anet_200"]
    board = Board(config.game["size"], 1)
    action_sequence = []
    while not board.finished:
        print("Player", board.player)
        board.pretty_state()
        if board.player == 1:
            action = agent.best_action(board.valid_actions, board.nn_state, verbose=True)
        else:
            action = int(input(board.valid_actions))
        action_sequence.append(action)
        board.play(action)
    board.pretty_state()
    if board.next_player == 1:
        print("Player 1 wins")
    else:
        print("Player 2 wins")


if __name__ == '__main__':
    import pickle
    # ts = pickle.load(open("buffers/3", "rb"))
    # print(len(ts))
    # play_manually()
    scores, detailed = tournament()
    for agent, score in detailed.items():
        print(f"{agent}:", score)

    print("\nTotals:")
    for agent, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{agent}:", score)
