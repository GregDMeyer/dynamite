
from itertools import combinations
from argparse import ArgumentParser
from sys import stderr
from numpy import random

from dynamite import config
from dynamite.operators import op_sum, op_product
from dynamite.extras import majorana
from dynamite.subspaces import Parity
from dynamite.states import State
from dynamite.tools import mpi_print, MPI_COMM_WORLD


def main():
    args = parse_args()

    # we print this to stderr to separate it from the data output below
    mpi_print('== Run parameters: ==', file=stderr)
    for key, value in vars(args).items():
        if key == 'seed':
            continue  # we handle seed below
        mpi_print(f'  {key}, {value}', file=stderr)

    # set a random seed, and output it for reproducibility
    if args.seed is None:
        seed = get_shared_seed()
    else:
        seed = args.seed
    mpi_print(f'  seed, {seed}', file=stderr)
    random.seed(seed)

    # extra newline for readability of the output
    mpi_print(file=stderr)

    # globally enable shell matrices (unless command line told us not to)
    config.shell = not args.no_shell

    # globally set the number of spins to ceil(N/2)
    config.L = (args.N+1)//2

    # ensures we get the same random numbers on all MPI ranks
    random.seed(args.seed)

    # the Hamiltonian conserves spin parity in the Z basis (Z2 symmetry)
    # but the majorana operators W and V take us between the two symmetry sectors
    # so we will make use of both parity subspaces (see below)
    even_space = Parity('even')
    odd_space = Parity('odd')

    W = majorana(0)
    V = majorana(1)

    # specify that these operators take odd to even and even to odd
    W.add_subspace(even_space, odd_space)
    W.add_subspace(odd_space, even_space)

    V.add_subspace(even_space, odd_space)
    V.add_subspace(odd_space, even_space)

    sorted_beta = sorted(args.b)

    # print the headers for the output CSV
    mpi_print("beta,t,C")

    for _ in range(args.H_iters):  # disorder realizations

        H = build_hamiltonian(args.N)

        # H conserves parity, and will operate on both odd and even spaces at various points
        # so we add both of them. which subspace is used will be automatically chosen based on the
        # state that H is applied to.
        # when only one argument is supplied to add_subspace, it is used for both the
        # "left" and "right" subspace (implying it is conserved)
        H.add_subspace(even_space)
        H.add_subspace(odd_space)

        for _ in range(args.state_iters):
            # we'll have psi start on the even subspace
            psi0 = State(state='random', subspace=even_space)
            psi1 = psi0.copy()  # a place to put the evolution result

            for i, b in enumerate(sorted_beta):
                # cost of time evolution is proportional to || bH ||
                # so, we can save some time by starting our imaginary time evolution at the previous
                # beta (note that the results will be correlated across beta values because we are
                # re-using psi, as long as we are careful in our analysis and do enough disorder
                # realizations this is OK)
                if i == 0:
                    delta_b = b
                else:
                    delta_b = b - sorted_beta[i-1]

                # do imaginary time evolution to compute e^{-beta/2 H} |psi>
                # we will take the expectation value of the OTOC with respect to the result
                H.evolve(psi0, t=-1j*delta_b, result=psi1)
                psi1.normalize()

                # set psi0 to equal psi1
                psi1.copy(result=psi0)

                # compute_otoc will not touch psi1 (but it does modify psi0)
                for t in args.t:
                    result = compute_otoc(psi0, psi1, t, H, W, V)

                    mpi_print(f"{b},{t},{result}")

                    # restore psi0 to equal psi1 for the next iteration
                    psi1.copy(result=psi0)


def build_hamiltonian_simple(N):
    '''
    This function is the most straightforward way to generate the
    Hamiltonian, but it's not nearly as fast as build_hamiltonian
    below. We still include it as an example.
    '''
    H = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                for l in range(k+1, N):
                    Jijkl = random.uniform(-1, 1)
                    H += Jijkl*majorana(i)*majorana(j)*majorana(k)*majorana(l)

    return 6/N**3 * H


def build_hamiltonian(N):
    '''
    This function builds the SYK Hamiltonian, and is about 10 times faster
    than build_hamiltonian_simple
    '''

    # pre-compute all the majorana operators because we will re-use them many times
    majoranas = [majorana(i) for i in range(N)]

    # This is a Python generator, which produces the terms of
    # the Hamiltonian
    def gen_products(N):
        for idxs in combinations(range(N), 4):
            # computes the product of the four majoranas
            # faster than using several "*" operations because it does not create a new
            # intermediate operator object with each multiplication
            p = op_product(majoranas[idx] for idx in idxs)

            # random value is the same on each rank because we set the seed explicitly in main()
            Jijkl = random.uniform(-1, 1)

            # using scale() is faster than doing 'Jijkl*p' because it does not create
            # a new operator object, instead just scaling the coeffs of the existing one
            p.scale(Jijkl)

            yield p

    # op_sum iterates through the terms generated by gen_products, and sums
    # them more efficiently than doing 'H += term' for each term
    H = op_sum(gen_products(N))

    # global prefactor
    # again, using scale() is much more efficient than "return 6/N**3 * H"
    # because it doesn't create a copy of the whole gigantic Hamiltonian
    H.scale(6/N**3)

    return H


def compute_otoc(psi0, psi1, t, H, W, V):
    '''
    Computes the value
    C = 2*real(<psi1| W(t) V(0) W(t) V(0) |psi0>) + 0.5
    where W(t) = e^{iHt} W e^{-iHt}

    the contents of psi1 will not be modified; but psi0 will
    '''

    # apply V, allocating a new vector
    # ideally one would reuse the same vector across disorder realizations, but
    # the cost of reallocating this vector once per iteration is negligible compared
    # to the rest of the computation (and the memory gets freed each iteration when
    # the variable goes out of scope)
    tmp_odd_0 = V*psi0  # note that tmp_odd_0 is in the "odd" subspace

    # next up is e^{-iHt} in the definition of W(t)
    # here we are implicitly allocating another vector in the odd subspace
    tmp_odd_1 = H.evolve(tmp_odd_0, t=t)

    # apply W, taking us back into the even subspace
    # we can reuse psi0 here to save some memory
    W.dot(tmp_odd_1, result=psi0)

    # apply the e^{iHt} on the other side of W(t)
    tmp_even = H.evolve(psi0, t=-t)

    # now V takes us back to the odd subspace again
    # we can reuse tmp_odd_0
    V.dot(tmp_even, result=tmp_odd_0)

    # now e^{-iHt} on the right of our final W(t)
    H.evolve(tmp_odd_0, t=t, result=tmp_odd_1)

    # finally back to the even subspace for the last time
    W.dot(tmp_odd_1, result=psi0)

    # and the final (reverse) time evolution on the left of the last W(t)
    H.evolve(psi0, t=-t, result=tmp_even)

    # at last we take the inner product with psi1
    result = psi1.dot(tmp_even)

    # this is the value C(t) that we really care about---see README
    return 2*result.real + 0.5


def get_shared_seed():
    '''
    Generate a seed for the random number generator, that is shared by all MPI ranks
    '''
    from random import SystemRandom

    # get PETSc's MPI communicator object
    comm = MPI_COMM_WORLD()

    # have rank 0 pick a seed
    if comm.rank == 0:
        # get a hardware-random number from the system to use as a seed
        seed = SystemRandom().randrange(2**32)
    else:
        seed = None

    # if there is only one rank, don't need to do anything fancy
    # doing this before using mpi4py below allows us to avoid needing mpi4py installed
    # when we only use one rank
    if comm.size == 1:
        return seed

    # otherwise, we need to communicate the seed among the ranks, using mpi4py
    # so we convert to a full-fledged mpi4py communicator class
    comm = comm.tompi4py()

    # now broadcast from rank 0 to all other ranks
    seed = comm.bcast(seed, root=0)

    return seed


def parse_args():
    parser = ArgumentParser(description='Compute OTOCs for the SYK model.')

    parser.add_argument('-N', default=30, type=int, help='number of majoranas')
    parser.add_argument('-b', default=[0.5], type=lambda s: [float(x) for x in s.split(',')],
                        help='comma-separated list of values of beta')
    parser.add_argument('-t', default=[0.5], type=lambda s: [float(x) for x in s.split(',')],
                        help='comma-separated list of values of the time t')
    parser.add_argument('--H-iters', default=1, type=int,
                        help='number of Hamiltonian disorder realizations')
    parser.add_argument('--state-iters', default=1, type=int,
                        help='number of random states per Hamiltonian')

    parser.add_argument('-s', '--seed', type=int,
                        help='seed for random number generator. if omitted, a random '
                             'seed is chosen by querying system hardware randomness')

    parser.add_argument('--no-shell', action='store_true',
                        help='disable shell matrices (they are enabled by default)')

    return parser.parse_args()


if __name__ == '__main__':
    main()
