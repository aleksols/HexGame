class Node:
    def __init__(self, state, edge_action, player):
        self.wins = 0
        self.visits = 0
        self.player = player
        self.state = state
        self.children = []
        self.parent = None
        self.edge_action = edge_action  # The action applied to get to this node
        self.expanded = False

    def has_children(self):
        return len(self.children) > 0

    def expand(self, child_saps):
        for s, a in child_saps:
            node = Node(s, a, not self.player)
            if node in self.children:
                continue
            else:
                self.children.append(node)
                node.parent = self
                break
        if len(self.children) == len(child_saps):
            self.expanded = True
        return node

    def delete_children(self, keep=None):
        for child in self.children:
            if child == keep:
                continue
            child.delete_children()
            del child


    def N(self):
        return self.visits

    @property
    def value(self):
        return self.wins / self.visits

    def pretty_print_values(self):
        print(self.wins, "/", self.visits)

    def __eq__(self, other):
        return self.state == other.state and \
               self.edge_action == other.edge_action and \
               self.player == self.player

    def __str__(self):
        return f"State: {self.state}, player: {self.player}"
