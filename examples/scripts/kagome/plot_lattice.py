
from matplotlib import pyplot as plt
import numpy as np


def plot_lattice(vertices, edges):

    f, ax = plt.subplots()

    # set up the plot
    ax.set_aspect('equal')
    ax.set_axis_off()

    ax.set_xlim(
        -0.5 + min(x for x, _ in vertices),
        0.5 + max(x for x, _ in vertices)
    )
    ax.set_ylim(
        -0.5 + min(y for _, y in vertices),
        0.5 + max(y for _, y in vertices)
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
            fontweight='bold',
            size=size_to_pts(0.18, f, ax)
        )

    for i, j in edges:
        neighbors = are_neighbors(vertices[i], vertices[j])
        color = '0.5' if neighbors else '0.8'
        plt.plot(*zip(vertices[i], vertices[j]),
                 color=color,
                 linewidth=bar_width,
                 zorder=0 if neighbors else -1)

    plt.show()


def are_neighbors(a, b):
    return np.isclose(np.hypot(b[0]-a[0], b[1]-a[1]), 1)


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
    from lattice_library import basis_to_graph, kagome_clusters

    if len(argv) < 2:
        lattice_name = '12'
    else:
        lattice_name = argv[1]

    plot_lattice(
        *basis_to_graph(kagome_clusters[lattice_name])
    )


if __name__ == '__main__':
    main()
