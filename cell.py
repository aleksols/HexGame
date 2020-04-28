class Cell:

    def __init__(self, index, pos=(0, 0), row=0, column=0):
        self.neighbours = []
        self.index = index
        self.pos = pos
        self.row = row
        self.column = column

        self.player = 0

    def add_neighbour(self, *args):
        for cell in args:
            if cell in self.neighbours:
                continue
            self.neighbours.append(cell)
            cell.neighbours.append(self)

    @property
    def color(self):
        if self.player == 0:
            return "white"
        if self.player == 1:
            return "red"
        return "black"

    @property
    def state(self):
        return self.player

    @property
    def nn_state(self):
        if self.player == 0:
            return [0, 0]  # Empty cell
        if self.player == 1:
            return [0, 1]  # Player 1 occupies the cell
        return [1, 0]  # Player 2 occupies the cell

    def occupy(self, player):
        self.player = player

    def clear(self):
        self.player = 0

    def __hash__(self):
        return hash(self.index) + hash(self.player)

    def __str__(self):
        return str(self.index)

    def __repr__(self):
        return f"<Cell {self.index}>"
