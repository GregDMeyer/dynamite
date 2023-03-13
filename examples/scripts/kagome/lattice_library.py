
import numpy as np


# The Kagome clusters listed below are from Table 1 of Lauchli et al. 2011
# [https://doi.org/10.1103/PhysRevB.83.212401], except 48, which is from
# Lauchli et al. 2019 [https://doi.org/10.1103/PhysRevB.100.155142]

kagome_clusters = {
    '12':  [( 2,  0), ( 0,  2)],
    '15':  [( 2, -1), (-1,  3)],
    '18a': [( 2, -1), ( 0,  3)],
    '18b': [( 2, -2), (-2, -1)],
    '21':  [( 2,  1), (-1,  3)],
    '24':  [( 1,  2), (-3,  2)],
    '27a': [( 2,  1), (-3,  3)],
    '27b': [( 3,  0), ( 0,  3)],
    '30':  [( 2,  1), (-2,  4)],
    '33':  [( 1,  2), ( 4, -3)],
    '36a': [(-2,  3), ( 4,  0)],
    '36b': [( 3,  0), (-3,  4)],
    '36c': [( 3,  0), (-1,  4)],
    '36d': [( 4, -2), (-2,  4)],
    '39a': [(-1,  3), ( 5, -2)],
    '39b': [( 1,  3), (-3,  4)],
    '42a': [(-1,  3), ( 5, -1)],
    '42b': [(-2,  4), ( 4, -1)],
    '48':  [( 4,  0), ( 0,  4)]
}


def basis_to_graph(basis, start_vertex=None):
    '''
    Given an pair of basis vectors defining a tiling of the Kagome lattice on a torus,
    number the vertices in one such tile and return a list of real-space coordinates
    for each vertex, as well as a list of (nearest-neighbor) edges between these vertices.

    start_vertex is the coordinates of the vertex from which to start the
    breadth-first search for numbering the vertices.
    '''

    if start_vertex is None:
        start_vertex = (0, 0)

    if not _is_lattice_point(start_vertex):
        raise ValueError('start point does not correspond to a vertex')

    neighbor_deltas = [
        (0, +1),
        (+1, 0),
        (+1, -1),
        (0, -1),
        (-1, 0),
        (-1, +1)
    ]

    vertices = [start_vertex]
    edges = set()
    pointer = 0

    while pointer < len(vertices):
        cur = vertices[pointer]

        for delta in neighbor_deltas:
            neighbor = tuple(v+d for v, d in zip(cur, delta))

            # wrap around torus if necessary
            neighbor = _translate_point_into_tile(neighbor, basis)

            if _is_lattice_point(neighbor):
                if neighbor not in vertices:
                    neighbor_idx = len(vertices)
                    vertices.append(neighbor)
                else:
                    neighbor_idx = vertices.index(neighbor)

                # only add each edge once
                if pointer < neighbor_idx:
                    edges.add((pointer, vertices.index(neighbor)))

        pointer += 1

    # finally transform the vertices to real space coordinates
    vertices = [(x + y/2, np.sqrt(3)*y/2) for x, y in vertices]

    return vertices, edges


def _is_lattice_point(point):
    return point[0] % 2 == 0 or point[1] % 2 == 1


def _translate_point_into_tile(point, basis_vecs):
    # Lauchli et al use unit cells of length 2, here it will be more convenient to just use units
    a, b = [np.array([2*v[0], 2*v[1]]) for v in basis_vecs]

    # "a" should always be the "more clockwise" one
    orientation = _loop_direction(a, (0, 0), b)
    if orientation == -1:
        a, b = b, a
    elif orientation == 0:
        raise ValueError('basis vectors are linearly dependent')

    origin = np.array([0, 0])
    far_corner = a + b

    rtn = np.array(point)

    if _loop_direction(a, origin, point) == -1:
        # point is below the bottom line
        rtn += b
    elif _loop_direction(b, far_corner, point) != +1:
        # point is on or above the upper line
        rtn -= b

    if _loop_direction(origin, b, point) == -1:
        # point is to the left of the left boundary
        rtn += a
    elif _loop_direction(far_corner, a, point) != +1:
        # point is on or to the right of the right boundary
        rtn -= a

    return tuple(rtn)


def _loop_direction(a, b, c):
    '''
    return whether the points form a clockwise loop (+1), a counter-clockwise loop (-1),
    or a line (0).
    '''
    v1 = (a[0]-b[0], a[1]-b[1])
    v2 = (c[0]-b[0], c[1]-b[1])
    cross_product = v1[0]*v2[1] - v1[1]*v2[0]

    # return sign of cross product
    if cross_product != 0:
        return int(cross_product//abs(cross_product))
    else:
        return 0


def _test():
    '''
    basic checks---just that each graph contains the correct number of spins, and that each
    vertex has degree exactly 4.
    '''
    from collections import defaultdict

    for cluster_name, basis in kagome_clusters.items():
        L = int(cluster_name[:2])

        vertices, edges = basis_to_graph(basis)

        if len(vertices) != L:
            raise ValueError(f'{len(vertices)} vertices for cluster "{cluster_name}"')

        assert max(max(a, b) for a, b in edges) == L-1

        d = defaultdict(lambda: 0)
        for e in edges:
            for v in e:
                d[v] += 1

        for v in range(L):
            if d[v] != 4:
                raise ValueError(f'vertex {v} in cluster "{cluster_name}" has degree {d[v]}')


if __name__ == '__main__':
    _test()
