import math
import numpy as np
from cell import Cell
import torch


class Board:

    def __init__(self, size, starting_player):
        self.cells = []
        self.size = size
        self.coord_repr = {}  # A coordinate based representation of cells
        self.init_cells_v2()
        self.starting_player = starting_player
        self.player = starting_player
        self.stones_places = 0
        self.p1_cells = {"rows": set()}
        self.p2_cells = {"columns": set()}

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
        self.player = state[0]
        for i, s in enumerate(state[1:]):
            self.cells[i].occupy(s)

    def reset(self):
        for cell in self.cells:
            cell.clear()
        self.player = self.starting_player

    def generate_child_states(self):
        child_saps = []
        for action in self.valid_actions:
            current_state = self.state
            current_state[0] = self.next_player
            current_state[action + 1] = self.player
            child_saps.append((current_state, action))
        return child_saps

    def play(self, action):
        if action not in self.valid_actions:
            raise ValueError(f"{action} is not a legal action i state {self.state}")
        chosen_cell = self.cells[action]
        chosen_cell.occupy(self.player)
        self.player = self.next_player

    def get_neighbour_pairs(self):
        pairs = []
        finished = []
        for cell in self.cells:
            for neighbour in cell.neighbours:
                if neighbour in finished:
                    continue
                pairs.append((cell, neighbour))
            finished.append(cell)
        return pairs

    @property
    def next_player(self):
        if self.player == 1:
            return 2
        return 1

    @property
    def valid_actions(self):
        return [i for i, cell in enumerate(self.cells) if cell.player == 0]

    @property
    def state(self):
        return [self.player] + [cell.state for cell in self.cells]

    @property
    def nn_state(self):
        state = self.state
        nn_state = []
        for s in state:
            if s == 1:
                nn_state += [0, 1]
            elif s == 2:
                nn_state += [1, 0]
            else:
                nn_state += [0, 0]
        return nn_state

    def pretty_state(self):
        for i in range(0, len(self.cells), self.size):
            print([c.state for c in self.cells[i: i + self.size]])

    @property
    def finished(self):
        # Previous and next player are the same, only the previous player could have won
        finished_player = self.next_player
        return self._player_one_finished() if finished_player == 1 else self._player_two_finished()

    def _player_one_finished(self):
        p1_cells = set()  # Use set because the "in" function is O(1) average for set and O(n) average for list
        rows = set()
        frontier = []
        for cell in filter(lambda x: x.player == 1, self.cells):
            p1_cells.add(cell)
            rows.add(cell.row)
            if cell.row == 0:
                frontier.append(cell)

        # Hopefully this will reduce runtime
        for row in range(self.size):
            if row not in rows:
                return False

        closed = set()
        while frontier:
            cell = frontier.pop(0)
            for neighbour in cell.neighbours:
                if neighbour in p1_cells and neighbour not in closed:
                    frontier.append(neighbour)
                    if neighbour.row == self.size - 1:
                        # print("p1 win")
                        return True
            closed.add(cell)
        return False

    def _player_two_finished(self):
        p2_cells = set()  # Use set because the "in" function is O(1) average for set and O(n) average for list
        columns = set()
        frontier = []
        for cell in filter(lambda x: x.player == 2, self.cells):
            p2_cells.add(cell)
            columns.add(cell.column)
            if cell.column == 0:
                frontier.append(cell)

        # Hopefully this will reduce runtime
        for col in range(self.size):
            if col not in columns:
                return False

        closed = set()
        while frontier:
            cell = frontier.pop(0)
            for neighbour in cell.neighbours:
                if neighbour in p2_cells and neighbour not in closed:
                    frontier.append(neighbour)
                    if neighbour.column == self.size - 1:
                        # print("p2 win")
                        return True
            closed.add(cell)
        return False


class ResBoard(Board):
    @property
    def nn_state(self):
        # state = torch.zeros(5, 5, 5)
        # state = torch.zeros(5, self.size, self.size)
        state = [[[0 for _ in range(self.size)] for _ in range(self.size)] for _ in range(5)]
        # state = [[[0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0]],
        #
        #          [[0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0]],
        #
        #          [[0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0]],
        #
        #          [[0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0]],
        #
        #          [[0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0]]]
        for cell in self.cells:
            state[2 - cell.player][cell.row][cell.column] = 1
        for r in range(self.size):
            for c in range(self.size):
                state[self.player + 2][r][c] = 1
        return torch.tensor(state, dtype=torch.float)


if __name__ == '__main__':
    import time

    start = time.time()
    b = ResBoard(6, 1)
    s1 = b.state
    print("State", b.state)
    import random

    # print(b.nn_state.size())
    for i in range(1000):
        b.set_state(s1)
        while not b.finished:
            nn_state = b.nn_state
            b.play(random.choice(b.valid_actions))

        b.reset()
    print(time.time() - start)
    start = time.time()
    for _ in range(1000):
        state = b.nn_state
        del state
    end = time.time() - start
    print(end)
    start = time.time()
    for _ in range(1000):
        state = b.finished
        del state
    end = time.time() - start
    print(end)
    state = b.state
    start = time.time()
    for _ in range(1000):
        b.set_state(state)
    end = time.time() - start
    print(end)
