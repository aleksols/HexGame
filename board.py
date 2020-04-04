import math

from cell import Cell


class Board:

    def __init__(self, size):
        self.cells = []
        self.size = size
        self.init_cells()


    def init_cells(self):
        counter = 0
        for row in range(0, self.size):
            new_cells = []
            for column in range(0, self.size):

                #  See rotation matrix on Wikipedia if ever in doubt about this calculation
                angle = -math.pi / 4
                x_pos = column
                y_pos = -row
                x = x_pos * math.cos(angle) - y_pos * math.sin(angle)
                y = x_pos * math.sin(angle) + y_pos * math.cos(angle)
                cell = Cell(counter, pos=(x, y), row=row, column=column)

                left_neighbour = new_cells[-1:]
                top_neighbours = []
                if row != 0:
                    top_neighbours = self.cells[row - 1][column: column + 2]

                neighbours = top_neighbours + left_neighbour

                cell.add_neighbour(*neighbours)
                new_cells.append(cell)
                counter += 1

            self.cells.append(new_cells)

    def set_state(self, state):
        i = 0
        for row in self.cells:
            for cell in row:
                cell.set_state(state[i])
                i += 1

    def reset(self):
        for row in self.cells:
            for cell in row:
                cell.player = None

    @property
    def valid_actions(self):
        actions = []
        for row in self.cells:
            for cell in row:
                if cell.player is None:
                    actions.append(cell.index)
        return actions

    @property
    def state(self):
        state = []
        for row in self.cells:
            for cell in row:
                state.append(cell.state)
        return state



if __name__ == '__main__':
    b = Board(4)
    print(b.state)
    print(b.valid_actions)
    b.set_state([(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1)])
    print(b.state)
    print(b.valid_actions)