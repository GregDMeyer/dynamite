
from argparse import ArgumentParser
from glob import glob
from os.path import join
from os import remove
from sys import stderr

from dynamite import config
from dynamite.operators import sigmax, sigmay, sigmaz, index_sum, index_product, op_sum
from dynamite.states import State
from dynamite.computations import entanglement_entropy
from dynamite.tools import mpi_print, MPI_COMM_WORLD


# TODO: what if it is killed while checkpointing?
# (seems unlikely but could happen...)


def main():
    args = parse_args()

    # print this to stderr to separate it from the data output below
    mpi_print('== Run parameters: ==', file=stderr)
    for key, value in vars(args).items():
        mpi_print(f'  {key}, {value}', file=stderr)
    mpi_print(file=stderr)  # an extra newline

    config.L = args.L
    config.shell = args.shell

    if args.checkpoint_every != 0:
        cycle_start, state = load_checkpoint(args.checkpoint_path)
    else:
        cycle_start = 0
        state = None

    if state is None:
        state = State(state=domain_wall_state_str(args.initial_state_dwalls, args.L))

    H = build_hamiltonian(args.alpha, 1, args.Jx, args.h_vec)

    # pi pulse operator
    X = index_product(sigmax())

    # the averaged "effective" Hamiltonian
    # (we just literally take the average of H and H conjugated by the pi pulse X)
    Deff = (H + X*H*X)/2

    # we create Deff and the Sz operators before the iterations start so that we
    # are not rebuilding the matrices for them every iteration
    Sz_ops = [0.5*sigmaz(i) for i in range(args.L)]

    # a workspace vector to store the output of the evolution in
    tmp = state.copy()

    # output the statistics at t=0
    if cycle_start == 0:
        print_stats(state, 0, tmp, Deff, Sz_ops)

    for cycle in range(cycle_start+1, args.n_cycles+1):
        H.evolve(state, result=tmp, t=args.T)
        X.dot(tmp, result=state)  # apply the pi pulse
        print_stats(state, cycle*args.T, tmp, Deff, Sz_ops)

        if args.checkpoint_every != 0 and cycle % args.checkpoint_every == 0:
            state.save(join(args.checkpoint_path, f'floquet_cycle_{cycle}'))

            # remove previous checkpoint, now that we have the new one
            if MPI_COMM_WORLD().rank == 0:
                if cycle > args.checkpoint_every:
                    prev_cycle = cycle-args.checkpoint_every
                    to_remove = glob(join(args.checkpoint_path, f'floquet_cycle_{prev_cycle}*'))
                    for fname in to_remove:
                        remove(fname)


def build_hamiltonian(alpha, Jz, Jx, h):
    # sums over all ranges of interaction
    # index_sum takes the interaction sigmaz(0)*sigmaz(r) and
    # translates it along the spin chain
    long_range_ZZ = op_sum(
        1/r**alpha * index_sum(0.25*sigmaz(0)*sigmaz(r))
        for r in range(1, config.L)
    )

    # an XX interaction on every neighboring pair of sites
    # the 0.25 is because spin operators are 1/2 times the Pauli
    nearest_neighbor_XX = index_sum(0.25*sigmax(0)*sigmax(1))

    # op_sum combines the three components of the magnetic field vector, and then
    # index_sum translates the resulting operator to every site along the spin chain
    magnetic_field = index_sum(
        op_sum(hi*0.5*s() for hi, s in zip(h, [sigmax, sigmay, sigmaz]))
    )

    return Jz*long_range_ZZ + Jx*nearest_neighbor_XX + magnetic_field


def print_stats(state, t, tmp, Deff, Sz_ops):
    '''
    Print out statistics about the state in CSV format. Also prints the CSV headers
    if t=0.
    '''
    if t == 0:
        mpi_print('t,Deff_energy,entropy,'+','.join(f'Sz{i}' for i in range(config.L)))

    # pass in tmp to avoid unnecessarily allocating a new vector here
    # probably doesn't actually matter that much for performance, but might as well
    Deff_energy = Deff.expectation(state, tmp_state=tmp)

    # half-chain entanglement entropy
    # NOTE: entanglement_entropy returns the EE value only on MPI rank 0, and -1 on all other ranks.
    #       this is OK here because mpi_print below only prints on rank 0
    entropy = entanglement_entropy(state, keep=range(config.L//2))

    # Sz expectation values for each spin
    Sz_vals = []
    for i in range(config.L):
        Sz_vals.append(Sz_ops[i].expectation(state, tmp_state=tmp))

    mpi_print(t, Deff_energy, entropy, *Sz_vals, sep=',')


def domain_wall_state_str(dwalls, L):
    '''
    Create a string like 'UUUUDDDDUUUU' that specifies a state with 'dwalls'
    domain walls.
    '''
    if dwalls >= L:
        raise ValueError('cannot have more domain walls than the number of spins - 1')

    c = 'U'
    rtn = ''
    for domain_idx in range(dwalls+1):
        rtn += c*((L-len(rtn)) // (dwalls-domain_idx+1))
        c = 'D' if c == 'U' else 'U'  # switch between 'D' and 'U'
    return rtn


def load_checkpoint(path):
    '''
    Load the checkpoint at path, if there is one there. Returns the next cycle number
    and the state object, or 0 and None if no checkpoint file was found.
    '''
    fnames = glob('floquet_cycle_*.vec', root_dir=path)
    if not fnames:
        return 0, None
    if len(fnames) > 1:
        raise RuntimeError("multiple checkpoint files found")

    fname = fnames[0]

    # extract the cycle number by trimming off the prefix and suffix
    cycle = int(fname[len('floquet_cycle_'):-len('.vec')])

    # path for from_file does not include extension
    state = State.from_file(join(path, fname[:-len('.vec')]))

    return cycle+1, state


def parse_args():
    parser = ArgumentParser(description='Evolve under a Floquet Hamiltonian')

    parser.add_argument('-L', type=int, default=14, help='number of spins')

    parser.add_argument('--Jx', type=float, default=0.19, help='coefficient on the XX term')
    parser.add_argument('--h-vec', type=lambda s: [float(x) for x in s.split(',')],
                        default=[0.21, 0.17, 0.13], help='magnetic field vector')
    parser.add_argument('--alpha', type=float, default=1.25,
                        help='power law for long range ZZ interaction')

    parser.add_argument('-T', type=float, default=0.12,
                        help='Floquet period')

    parser.add_argument('--initial-state-dwalls', type=int, default=1,
                        help='Number of domain walls to include in initial product state')

    parser.add_argument('--n-cycles', type=int, default=int(1e4),
                        help='Total number of Floquet cycles')

    parser.add_argument('--checkpoint-path', default='./',
                        help='where to save the state vector for checkpointing/restarting. '
                             '[default: ./]')
    parser.add_argument('--checkpoint-every', default=0, type=int,
                        help='how frequently to save checkpoints, in number of cycles. '
                             'if this option is omitted, checkpoints will not be saved.')

    parser.add_argument('--shell', action='store_true', help='use matrix-free computation')

    args = parser.parse_args()

    if len(args.h_vec) != 3:
        raise ValueError('command-line value for -h must be exactly three comma-separated numbers')

    return args


if __name__ == '__main__':
    main()
