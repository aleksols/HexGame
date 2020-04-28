from mcts_kernel import MCTS

class MCTSResNet(MCTS):
    def __init__(self, board, anet, c=1):
        super().__init__(board, c, anet)