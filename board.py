import math
import numpy as np
from cell import Cell


class Board:

    def __init__(self, size, starting_player):
        self.cells = []
        self.size = size
        self.coord_repr = {}  # A coordinate based representation of cells
        self.init_cells_v2()

        self.player = starting_player
        self.stones_places = 0

    def init_cells_v2(self):
        col_vector = np.array([1, -1])
        row_vector = np.array([-1, -1])

        # Adding cells
        for i in range(self.size ** 2):
            col = i % self.size
            row = i // self.size
            position = col * col_vector + row * row_vector
            cell = Cell(i, pos=(position[0], position[1]), row=row, column=col)
            self.cells.append(cell)
            self.coord_repr[(row, col)] = cell

        # Adding neighbours
        for cell in self.cells:
            r = cell.row
            c = cell.column
            neighbour_coord = [(r - 1, c), (r - 1, c + 1), (r, c + 1), (r + 1, c), (r + 1, c - 1), (r, c - 1)]
            for coord in neighbour_coord:
                if coord in self.coord_repr.keys():
                    cell.add_neighbour(self.coord_repr[coord])

    def set_state(self, state):
        for i, s in enumerate(state):
            self.cells[i].set_state(s)

    def reset(self):
        for cell in self.cells:
            cell.player = None

    def play(self, action):
        # TODO implement
        return None

    @property
    def valid_actions(self):
        # TODO maybe move to state manager
        return [i if cell.player is None else 2 for i, cell in enumerate(self.cells)]

    @property
    def state(self):
        # TODO implement player into state and use in play as well
        return self.cells.copy()

    @property
    def final_state(self):
        return True

if __name__ == '__main__':
    b = Board(4)
    print(b.valid_actions)
    b.set_state([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
