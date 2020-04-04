class Cell:

    def __init__(self, index, pos=(0, 0), row=0, column=0):
        self.neighbours = []
        self.index = index
        self.pos = pos
        self.row = row
        self.column = column

        self.player = None

    def add_neighbour(self, *args):
        for cell in args:
            if cell in self.neighbours:
                continue
            self.neighbours.append(cell)
            cell.neighbours.append(self)

    @property
    def color(self):
        if self.player is None:
            return "white"
        if self.player is True:
            return "black"
        return "red"

    @property
    def state(self):
        if self.player is None:
            return (0, 0)  # Empty cell
        if self.player is True:
            return (0, 1)  # Player 1 occupies the cell
        return (1, 0)  # Player 2 occupies the cell

    def set_state(self, state):
        if state[0]:
            self.player = False
        elif state[1]:
            self.player = True
        else:
            self.player = None

    def __hash__(self):
        return self.index

    def __str__(self):
        return str(self.index)

    def __repr__(self):
        return f"<Cell {self.index}>"
