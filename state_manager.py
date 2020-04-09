class StateManager:
    def __init__(self, board):
        self.game_state = None
        self.player = None
        self.board = board

    def generate_child_states(self):
        pass

    def set_state(self, state, player):
        pass

    def apply_action(self, action):
        pass

    @property
    def finished(self):
        pass