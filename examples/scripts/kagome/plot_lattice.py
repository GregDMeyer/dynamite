
from matplotlib import pyplot as plt
import numpy as np


def plot_lattice(vertices, edges):

    f, ax = plt.subplots()

    # set up the plot
    ax.set_aspect(np.sqrt(3))
    ax.set_axis_off()

    ax.set_xlim(
        -0.5,
        0.5 + max(row for row, _ in vertices)
    )
    ax.set_ylim(
        -0.5,
        0.5 + max(col for _, col in vertices)
    )

    vertex_width = size_to_pts(5, f, ax)
    bar_width = size_to_pts(0.1, f, ax)

    plt.scatter(
        *zip(*vertices),
        s=vertex_width,
        c='white',
        edgecolor='black',
        linewidth=1.5
    )
    for idx, vertex in enumerate(vertices):
        plt.text(
            *vertex, str(idx),
            ha='center', va='center_baseline',
            fontweight='bold'
        )

    done = set() # check for doubles
    for i, j in edges:
        plt.plot(*zip(vertices[i], vertices[j]),
                 color='red' if (i,j) in done else '0.5',
                 linewidth=bar_width,
                 zorder=0)
        done.add((i,j))

    plt.show()


def size_to_pts(size, fig, ax):
    '''
    Convert a size in data units to a number of points, for use in
    e.g. linewidth argument
    '''
    ppd = 72./fig.dpi
    trans = ax.transData.transform
    return ((trans((size, 1))-trans((0, 0)))*ppd)[0]


def main():
    from sys import argv
    from lattice_library import parse_diagram, lattice_diagrams

    if len(argv) < 2:
        lattice = 'L12'
    else:
        lattice = argv[1]

    plot_lattice(*parse_diagram(lattice_diagrams[lattice]))


if __name__ == '__main__':
    main()
