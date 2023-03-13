
import numpy as np


lattice_diagrams = {
######
'L6' : '''
 o o
o   o
 o o
''',
######
'L12' : '''
   o
o o o o
 o   o
o o o o
   o
'''
}


def parse_diagram(diagram, start_vertex=None):
    '''
    Given a string denoting a subset of vertices of the triangular lattice (see examples),
    number the vertices and return a list of the indices and their coordinates,
    as well as a list of edges (for nearest-neighbor edges).

    Coordinates are integers corresponding to the row and column of the characters
    in the string, with (0,0) being the *bottom left* of the non-blank rows.

    start_vertex is the coordinates of the vertex from which to start the
    breadth-first search for numbering the vertices.
    '''
    rows = diagram.split('\n')

    # index rows from the bottom (so we can use x, y coords)
    rows = rows[::-1]

    # remove blank lines before and after
    while all(c == ' ' for c in rows[0]):
        rows = rows[1:]

    while all(c == ' ' for c in rows[-1]):
        rows = rows[:-1]

    # use bottom left vertex by default
    if start_vertex is None:
        start_vertex = (rows[0].index('o'), 0)

    return breadth_first_search_rows(rows, start_vertex)


def breadth_first_search_rows(rows, start_vertex):
    col, row = start_vertex
    if rows[row][col] == ' ':
        raise ValueError('start point does not correspond to a vertex')

    neighbor_deltas = [
        (-1, -1),
        (+1, -1),
        (+2, 0),
        (+1, +1),
        (-1, +1),
        (-2, 0)
    ]

    vertices = [start_vertex]
    edges = set()
    pointer = 0

    while pointer < len(vertices):
        cur_vertex = vertices[pointer]

        for delta in neighbor_deltas:
            new_vertex = tuple(v+d for v, d in zip(cur_vertex, delta))
            new_col, new_row = new_vertex

            if not ((0 <= new_row < len(rows)) and (0 <= new_col < len(rows[new_row]))):
                continue

            if rows[new_row][new_col] != ' ':
                if new_vertex not in vertices:
                    vertices.append(new_vertex)
                edges.add((pointer, vertices.index(new_vertex)))

        pointer += 1

    if len(vertices) != sum(len(row)-row.count(' ') for row in rows):
        raise ValueError('not all vertices were found via breadth-first search'
                         '---does your diagram contain disconnected sets?')

    return vertices, edges


def print_index_map(vertices):
    n_rows = max(row for _, row in vertices)+1
    for row in range(n_rows-1, -1, -1):  # start at the top
        cols = max(col for col, crow in vertices if crow == row)+1
        col = 0
        while col < cols:
            try:
                idx = vertices.index((col, row))
            except ValueError:
                print(' ', end='')
                col += 1
            else:
                print(idx, end='')
                col += min(len(str(idx)), 2)

        print()


def diagram_from_basis(a, b):
    # "a" should always be the one clockwise
    if cross_product(a, b) < 0:
        a, b = b, a

    # they use unit cells of length 2, here it will be more convenient to just use units
    a = (2*a[0], 2*a[1])
    b = (2*b[0], 2*b[1])

    len_x = abs(a[0]) + abs(b[0])
    len_y = abs(a[1]) + abs(b[1])

    origin_x = abs(min(0, a[0], b[0]))
    origin_y = abs(min(0, a[1], b[1]))

    lattice = np.ones((len_x+1, len_y+1))

    # turn it from a triangular lattice into kagome
    lattice[1::2, 0::2] = 0

    # now delete everything that is outside of the parallelogram
    # specified by a and b
    corners = [np.array(v) for v in [(0, 0), a, (a[0]+b[0], a[1]+b[1]), b]]

    for x in range(len_x+1):
        for y in range(len_y+1):
            # check that point is "on the correct side" of each line
            for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                side_vec = corners[j] - corners[i]
                point_vec = np.array([x-origin_x, y-origin_y]) - corners[i]
                if cross_product(side_vec, point_vec) < 0:
                    lattice[x, y] = 0

    diagram_rows = []
    for y in range(len_y+1):
        diagram_rows.append(y*' ')
        for x in range(len_x+1):
            if lattice[x, y]:
                diagram_rows[-1] += 'o '
            else:
                diagram_rows[-1] += '  '

    # get rid of leading spaces, and form a single string
    to_trim = min(len(s) - len(s.lstrip(' ')) for s in diagram_rows)
    rtn = '\n'.join(row[to_trim:] for row in diagram_rows[::-1])

    return rtn


def cross_product(a, b):
    return a[0]*b[1] - a[1]*b[0]


def main():
    v, _ = parse_diagram(lattice_diagrams['L12'])
    print_index_map(v)


if __name__ == '__main__':
    main()
