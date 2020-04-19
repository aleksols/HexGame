from board import Board
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import config

matplotlib.use("TkAgg")

def visualize(board, action_sequence):
    print(action_sequence)
    board.reset()
    fig, board_ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 6)

    board_ax.axes.get_xaxis().set_visible(True)
    board_ax.axes.get_yaxis().set_visible(True)
    y_max = 0.5
    y_min = -((board.size - 1) * 2 ** 0.5 + 0.5)
    x_max = ((board.size - 1) * 2 ** 0.5) / 2 + 0.5
    x_min = -((board.size - 1) * 2 ** 0.5) / 2 - 0.5
    def animate(i):
        if i > len(action_sequence):
            return
        if i > 0:
            board.play(action_sequence[i - 1])

        board_ax.clear()
        board_ax.set_ylim(y_min, y_max)
        board_ax.set_xlim(x_min, x_max)
        plt.axis("scaled")
        edges = board.get_neighbour_pairs()
        for edge in edges:
            x_start = edge[0].pos[0]
            y_start = edge[0].pos[1]
            x_end = edge[1].pos[0]
            y_end = edge[1].pos[1]
            edge_size = 1
            edge_color = "k-"
            if edge[0].player == edge[1].player and edge[0].player is not None:
                edge_color = "red" if edge[0].player == 1 else "black"
                edge_size = 4
            board_ax.plot([x_start, x_end], [y_start, y_end], edge_color, linewidth=edge_size, zorder=1)  # Draw black lines between cells
        for cell in board.cells:
            board_ax.add_patch(
                matplotlib.patches.Circle(
                    xy=cell.pos,
                    radius=0.45,
                    facecolor=cell.color,
                    edgecolor="brown",
                    fill=True,
                    zorder=2
                )
            )
            text_color = "white" if cell.color == "black" else "black"
            board_ax.text(x=cell.pos[0], y=cell.pos[1], s=cell.index, fontsize=12, color=text_color)
        arrows = [
            [board.size - 1, 0, -0.3, -0.3],
            [-board.size + 1, 0, 0.3, -0.3],
            [board.size - 1, -2 * board.size + 2, -0.3, 0.3],
            [-board.size + 1, -2 * board.size + 2, 0.3, 0.3]
        ]
        arrow_colors = ["red", "black", "black", "red"]
        for arrow, color in zip(arrows, arrow_colors):
            board_ax.arrow(*arrow, head_width=0.1, head_length=0.1, fc=color, ec=color)

    ani = animation.FuncAnimation(fig, animate, interval=300, blit=False)
    plt.show()

if __name__ == '__main__':
    from board import Board
    for i in range(3, 11):
        b = Board(i, 1)
        a = range(i**2)
        visualize(b, a)