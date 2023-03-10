
from dynamite import config
from dynamite.operators import sigmax, sigmay, sigmaz, index_sum
from dynamite.subspaces import SpinConserve
from dynamite.computations import entanglement_entropy

import numpy as np
from argparse import ArgumentParser


def parse_args():
    '''
    Read arguments from the command line.
    '''
    parser = ArgumentParser()

    parser.add_argument('-L', type=int, required=True, help='spin chain length')
    parser.add_argument('--seed', type=int, default=0xB0BACAFE,
                        help='seed for random number generator')
    parser.add_argument('--iters', type=int, default=16,
                        help='number of disorder realizations')

    parser.add_argument('--energy-points', type=int, default=3,
                        help='number of points in the spectrum to target')
    parser.add_argument('--h-points', type=int, default=5,
                        help='number of disorder strengths to test')
    parser.add_argument('--h-min', type=float, default=1,
                        help='minimum value of h')
    parser.add_argument('--h-max', type=float, default=5,
                        help='maximum value of h')
    parser.add_argument('--nev', type=int, default=32,
                        help='number of eigenpairs to compute at each point')

    return parser.parse_args()


def build_hamiltonian(h, seed=0xB0BACAFE):
    '''
    Implements the nearest-neighbor Heisenberg interaction on a 1D spin chain,
    plus random Z fields on each site.
    '''

    # 0.25 because we are working with Paulis and want spin operators
    one_site_heisenberg = 0.25*sum(s(0)*s(1) for s in [sigmax, sigmay, sigmaz])
    full_chain_heisenberg = index_sum(one_site_heisenberg)

    # if you run in parallel with MPI you need to make sure your
    # random seeds are the same on each MPI rank!
    # I like to pass a seed on the command line to each
    # disorder realization
    np.random.seed(seed)

    # 0.5 again for Pauli -> spin operator conversion
    random_fields = sum(0.5*np.random.uniform(-h, h)*sigmaz(i) for i in range(config.L))

    return full_chain_heisenberg + random_fields


def print_eig_stats(evals, evecs, h, energy_point):
    '''
    Compute the mean adjacent gap ratio and half-chain entanglement entropy
    for the provided eigenvalues and eigenstates
    '''
    # sum the entropy for all evecs then divide by nev for the mean
    entropy = sum(entanglement_entropy(v, keep=range(config.L//2)) for v in evecs)
    entropy /= len(evecs)

    # compute the adjacent gap ratio of the eigenvals
    evals = sorted(evals)
    ratio = 0
    for i in range(1, len(evals)-1):
        this_gap = evals[i] - evals[i-1]
        next_gap = evals[i+1] - evals[i]
        ratio += min(this_gap, next_gap) / max(this_gap, next_gap)
    ratio /= len(evals)-2

    print(f'{h}, {energy_point}, {entropy}, {ratio}')


def main():
    args = parse_args()

    # set spin chain length globally for dynamite
    config.L = args.L

    # work in half-filling subspace
    config.subspace = SpinConserve(args.L, args.L//2)

    # column headers
    print('h,energy_point,entropy,ratio')

    seed = args.seed
    for _ in range(args.iters):
        # so we get a new seed each disorder realization
        seed = hash(seed)

        for h in np.linspace(args.h_min, args.h_max, args.h_points):
            H = build_hamiltonian(h, seed)

            # first solve for the exterior ones

            # by default eigsolve finds the lowest eigenpairs
            evals, evecs = H.eigsolve(nev=args.nev, getvecs=True)
            print_eig_stats(evals, evecs, h, energy_point=0)
            min_eval = evals[0]

            # now the highest ones
            evals, evecs = H.eigsolve(nev=args.nev, which='largest', getvecs=True)
            print_eig_stats(evals, evecs, h, energy_point=1)
            max_eval = evals[0]

            for energy_point in np.linspace(0, 1, args.energy_points)[1:-1]:
                energy_target = min_eval + energy_point*(max_eval-min_eval)
                evals, evecs = H.eigsolve(nev=args.nev, target=energy_target, getvecs=True)
                print_eig_stats(evals, evecs, h=h, energy_point=energy_point)


if __name__ == '__main__':
    main()
