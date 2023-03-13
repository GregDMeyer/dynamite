
from sys import argv
from argparse import ArgumentParser
from datetime import datetime

from dynamite.operators import sigmax, sigmay, sigmaz, op_sum
from dynamite.subspaces import SpinConserve, XParity
from dynamite.tools import mpi_print

from lattice_library import kagome_clusters, basis_to_graph


def heisenberg(i, j):
    '''
    The Heisenberg interaction between sites i and j.
    '''
    # 0.25 to account for spin operators instead of Paulis
    return op_sum(0.25*s(i)*s(j) for s in [sigmax, sigmay, sigmaz])


def build_hamiltonian(cluster_name):
    '''
    Build the nearest-neighbor Heisenberg interaction with J=1 for
    the Kagome lattice on a torus, specified by the clusters in
    lattice_library.py.
    '''
    _, edges = basis_to_graph(kagome_clusters[cluster_name])
    return op_sum(heisenberg(i, j) for i, j in edges)


def main():
    args = parse_args()

    H = build_hamiltonian(args.cluster)

    # number of spins in the support of H
    N = H.get_length()

    # total magnetization is conserved, plus an extra Z2 symmetry on top
    H.subspace = SpinConserve(N, N//2)

    # TODO: when can we also apply XParity? when L is even?

    H.shell = args.shell

    # time the eigsolve!
    tick = datetime.now()
    ground_state_energy = H.eigsolve()[0]
    tock = datetime.now()

    mpi_print(f'Ground state energy E: {ground_state_energy}')
    mpi_print(f'E/N: {ground_state_energy/N}')

    mpi_print(f'\nSolve completed in {tock-tick}')


def parse_args():
    parser = ArgumentParser(description='Solve for the ground state energy of the Heisenberg model '
                                        'on the Kagome lattice.')

    parser.add_argument('cluster', default='12', help='which Kagome cluster to use '
                                                      '(see lattice_library.py)')
    parser.add_argument('--shell', action='store_true', help='whether to use shell matrices')

    return parser.parse_args()


if __name__ == '__main__':
    main()
