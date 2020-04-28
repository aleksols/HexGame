import torch
from tqdm import tqdm

from config import TournamentConf, GameConf
from agent import Agent
from anet import ANET
from board import Board, ResBoard
from residual_anet import ResNet
from CnnAnet import ConvAnet
from visualize import visualize
from os import listdir


def get_agents(policy="probabilistic"):
    agents = {}
    directory = "models/" + TournamentConf.directory + "/"
    prefix = TournamentConf.file_prefix
    model_paths = [directory + f for f in filter(lambda x: x[:len(prefix)] == prefix, listdir(directory))]
    for path in model_paths:
        agents[path.replace(directory, "")] = Agent(ANET.load(path), policy)
    return agents


def play_single(agent1, agent2, starting_player=1, visual=False):
    # In case agent1 and agent2 uses different types of neural nets
    a1_board = Board(GameConf.size, starting_player)
    a2_board = Board(GameConf.size, starting_player)
    if isinstance(agent1.anet, ResNet) or isinstance(agent1.anet, ConvAnet):
        a1_board = ResBoard(GameConf.size, starting_player)
    if isinstance(agent2.anet, ResNet) or isinstance(agent2.anet, ConvAnet):
        a2_board = ResBoard(GameConf.size, starting_player)

    board = Board(GameConf.size, starting_player)
    action_sequence = []
    while not board.finished:
        a1_board.set_state(board.state)
        a2_board.set_state(board.state)
        if board.player == 1:
            action = agent1.best_action(board.valid_actions, a1_board.nn_state, verbose=False)
        else:
            action = agent2.best_action(board.valid_actions, a2_board.nn_state, verbose=False)
        action_sequence.append(action)
        board.play(action)
    if visual:
        visualize(board, action_sequence)
    if board.next_player == 1:
        return 1, 0, action_sequence  # agent1 points, agent2 points and action sequence
    return 0, 1, action_sequence


def tournament():
    agents = get_agents(TournamentConf.agent_policies)
    detailed = {}
    scores = {}
    for a in agents.keys():
        detailed[(a, "starting")] = 0
        detailed[(a, "not starting")] = 0
        scores[a] = 0

    num_games = TournamentConf.games
    player_ones = list(agents.keys())
    player_twos = list(agents.keys())

    for i, agent1 in enumerate(player_ones[:-1], 1):
        for agent2 in player_twos[i:]:
            if agent1 == agent2:
                continue
            for game in range(1, num_games + 1):
                visual = game in TournamentConf.visualize
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


def play_manually(path, style):
    agent = Agent(ANET.load(path), style)
    board = Board(GameConf.size, 1)
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


def benchmark():
    import pickle
    a1 = Agent(ANET.load("models/6/new_best_try_v2_6_anet_600"), "probabilistic")
    print("a1", a1.anet)
    a2 = Agent(pickle.load(open("working_models/new_best_try6_anet_600", "rb")), "probabilistic")
    print("a2", a2.anet)
    a1_score = 0
    a2_score = 0
    num_games = 1000
    for game in tqdm(range(1, num_games + 1)):
        res1, res2, seq = play_single(a1, a2, game % 2 + 1, visual=False)
        a1_score += res1
        a2_score += res2
    print("agent1:", a1_score, "agent2:", a2_score)


def print_results(scores, details):
    for agent, score in details.items():
        print(f"{agent}:", score)

    print("\nTotals:")
    for agent, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{agent}:", score)


if __name__ == '__main__':
    # benchmark()
    # play_manually("OHT/6_anet_600", "greedy")
    scores, details = tournament()
    print_results(scores, details)
