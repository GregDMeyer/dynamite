
import argparse as ap
from random import uniform,seed
from timeit import default_timer
from itertools import combinations
import numpy as np

from dynamite import config
from dynamite.states import State
from dynamite.operators import sigmax, sigmay, sigmaz
from dynamite.operators import op_sum, op_product, index_sum
from dynamite.extras import majorana
from dynamite.subspaces import Full, Parity, SpinConserve, Auto, XParity
from dynamite.tools import track_memory, get_max_memory_usage
from dynamite.computations import reduced_density_matrix


def parse_args(argv=None):

    parser = ap.ArgumentParser(description='Benchmarking test for dynamite.')

    parser.add_argument('-L', type=int, required=True,
                        help='Size of the spin chain.')

    parser.add_argument('-H', choices=['MBL', 'long_range', 'SYK', 'ising', 'XX', 'heisenberg'],
                        help='Hamiltonian to use')

    parser.add_argument('--shell', action='store_true',
                        help='Make a shell matrix instead of a regular matrix.')
    parser.add_argument('--no-precompute-diagonal', action='store_true',
                        help='Turn off precomputation of the matrix diagonal for shell matrices.')
    parser.add_argument('--gpu', action='store_true',
                        help='Run computations on GPU instead of CPU.')

    parser.add_argument('--slepc_args', type=str, default='',
                        help='Arguments to pass to SLEPc.')
    parser.add_argument('--track_memory', action='store_true',
                        help='Whether to compute max memory usage')

    parser.add_argument('--subspace', choices=['full', 'parity',
                                               'spinconserve',
                                               'auto', 'nosortauto'],
                        default='full',
                        help='Which subspace to use.')
    parser.add_argument('--which_space', type=str,
                        help='The particular subspace to use. For parity, options are "even" '
                        'and "odd", for spinconserve an integer number of up spins, for auto '
                        'supply a starting state like UUUUDDDD.')
    parser.add_argument('--xparity', choices=['plus', 'minus'], nargs='?', const='plus',
                        help='If provided, applies the XParity subspace of the specified sector '
                        '(+ if sector not specified) on top of the subspace specified with the '
                        '--subspace argument (or XParity alone if no other subspace was '
                        'specified).')

    parser.add_argument('--evolve', action='store_true',
                        help='Request that the Hamiltonian evolves a state.')
    parser.add_argument('-t', type=float, default=50.0,
                        help='The time to evolve for.')
    parser.add_argument('--no_normalize_t', action='store_true',
                        help='Turn off the default behavior of dividing the evolve time by the '
                             'matrix norm, which should yield a fairer comparison across models'
                             ' and system sizes.')

    parser.add_argument('--mult', action='store_true',
                        help='Simply multiply the Hamiltonian by a vector.')
    parser.add_argument('--mult_count', type=int, default=1,
                        help='Number of times to repeat the multiplication.')

    parser.add_argument('--norm', action='store_true',
                        help='Compute the norm of the matrix.')

    parser.add_argument('--eigsolve', action='store_true',
                        help='Request to solve for eigenvalues of the Hamiltonian.')
    parser.add_argument('--nev', type=int ,default=1,
                        help='The number of eigenpairs to solve for.')
    parser.add_argument('--target', type=float,
                        help='The target for a shift-invert eigensolve.')

    parser.add_argument('--rdm', action='store_true',
                        help='Compute a reduced density matrix')
    parser.add_argument('--keep', type=lambda s: [int(x) for x in s.split(',')],
                        help='A list of spins, separated by commas, to keep during the '
                        'RDM computation. By default, the first half are kept.')

    parser.add_argument('--check-conserves', action='store_true',
                        help='Benchmark the check for whether the given subspace is conserved by '
                             'the matrix.')

    args = parser.parse_args(argv)

    # we need the norm anyway for this; might as well benchmark it
    if args.evolve and not args.no_normalize_t:
        args.norm = True

    return args

def build_subspace(params, hamiltonian=None):
    space = params.which_space

    if params.subspace == 'full':
        rtn = Full()

    elif params.subspace == 'parity':
        if space is None:
            space = 'even'
        rtn = Parity(space)

    elif params.subspace == 'spinconserve':
        if space is None:
            space = params.L//2
        else:
            space = int(space)

        rtn = SpinConserve(params.L, space)

    elif params.subspace in ['auto', 'nosortauto']:
        if space is None:
            half_length = params.L // 2
            space = 'U'*half_length + 'D'*(params.L - half_length)
        rtn = Auto(hamiltonian, space, sort=params.subspace=='auto')

    else:
        raise ValueError("invalid subspace")

    if params.xparity is not None:
        sector = {'plus': '+', 'minus': '-'}[params.xparity]
        rtn = XParity(rtn, sector=sector)

    return rtn

def build_hamiltonian(params):

    if params.H == 'MBL':
        # dipolar interaction
        rtn = index_sum(op_sum(0.25*s(0)*s(1) for s in (sigmax, sigmay, sigmaz)))
        # quenched disorder in z direction
        seed(0)
        for i in range(params.L):
            rtn += uniform(-3, 3) * 0.5 * sigmaz(i)

    elif params.H == 'long_range':
        # long-range ZZ interaction
        rtn = op_sum(index_sum(0.25*sigmaz(0)*sigmaz(i)) for i in range(1, params.L))
        # nearest neighbor XX
        rtn += 0.5 * index_sum(0.25*sigmax(0)*sigmax(1))
        # some other fields
        rtn += sum(0.05*index_sum(s()) for s in [sigmax, sigmay, sigmaz])

    elif params.H == 'SYK':
        seed(0)

        # only compute the majoranas once
        majoranas = [majorana(i) for i in range(params.L*2)]

        def gen_products(L):
            for idxs in combinations(range(L*2), 4):
                p = op_product(majoranas[idx] for idx in idxs)
                p.scale(uniform(-1, 1))
                yield p

        rtn = op_sum(gen_products(params.L))
        rtn.scale(np.sqrt(6/(params.L*2)**3))

    elif params.H == 'ising':
        rtn = index_sum(0.25*sigmaz(0)*sigmaz(1)) + 0.1*index_sum(sigmax())

    elif params.H == 'XX':
        rtn = index_sum(0.25*sigmax(0)*sigmax(1))

    elif params.H == 'heisenberg':
        rtn = index_sum(op_sum(0.25*s(0)*s(1) for s in (sigmax, sigmay, sigmaz)))

    else:
        raise ValueError('Unrecognized Hamiltonian.')

    # conservation check can take a long time; we benchmark it separately
    # TODO: speed up CheckConserves and remove this
    rtn.allow_projection = True

    return rtn

def compute_norm(hamiltonian):
    return hamiltonian.infinity_norm()

def do_eigsolve(params, hamiltonian):
    hamiltonian.eigsolve(nev=params.nev,target=params.target)

def do_evolve(params, hamiltonian, state, result):
    # norm should be precomputed by now so the following shouldn't affect
    # the measured cost of time evolution
    t = params.t
    if not params.no_normalize_t:
        t /= hamiltonian.infinity_norm()
    hamiltonian.evolve(state, t=t, result=result)

def do_mult(params, hamiltonian, state, result):
    for _ in range(params.mult_count):
        hamiltonian.dot(state, result)

def do_rdm(state, keep):
    reduced_density_matrix(state, keep)

def do_check_conserves(hamiltonian):
    hamiltonian.conserves(hamiltonian.subspace)

# this decorator keeps track of and times function calls
def log_call(function, stat_dict, alt_name=None):
    config._initialize()
    from petsc4py.PETSc import Sys
    Print = Sys.Print

    if alt_name is None:
        fn_name = function.__name__
    else:
        fn_name = alt_name

    def rtn(*args, **kwargs):
        if __debug__:
            Print('beginning', fn_name)

        tick = default_timer()
        rtn_val = function(*args, **kwargs)
        tock = default_timer()

        if __debug__:
            Print('completed', fn_name)

        stat_dict[fn_name] = tock-tick

        return rtn_val

    return rtn

def main():
    main_start = default_timer()

    arg_params = parse_args()
    slepc_args = arg_params.slepc_args.split(' ')
    config.initialize(slepc_args, gpu=arg_params.gpu)
    config.L = arg_params.L
    config.shell = arg_params.shell

    from petsc4py.PETSc import Sys
    Print = Sys.Print

    if not __debug__:
        Print('---ARGUMENTS---')
        for k,v in vars(arg_params).items():
            Print(str(k)+','+str(v))

    if arg_params.track_memory:
        track_memory()

    stats = {}

    # build our Hamiltonian, if we need it
    if arg_params.H is not None:
        H = log_call(build_hamiltonian, stats)(arg_params)
    else:
        if (arg_params.subspace == 'auto' or
                any(getattr(arg_params, x) for x in ['norm', 'eigsolve', 'evolve', 'mult'])):
            raise ValueError('Must specify Hamiltonian for this benchmark.')
        H = None

    # build our subspace
    subspace = log_call(build_subspace, stats)(arg_params, H)
    if H is not None:
        H.subspace = subspace

        if arg_params.no_precompute_diagonal:
            H.precompute_diagonal = False

        Print('H statistics:')
        Print(' dim:', H.dim[0])
        Print(' nnz:', H.nnz)
        Print(' density:', H.density)
        Print(' nterms:', H.nterms)
        log_call(H.build_mat, stats)()

    # build some states to use in the computations
    if arg_params.evolve or arg_params.mult or arg_params.rdm:
        in_state = State(L=arg_params.L, subspace=subspace)
        out_state = State(L=arg_params.L, subspace=subspace)
        log_call(in_state.set_random, stats, alt_name="set_random_state")()
    else:
        in_state = out_state = None

    # compute the norm
    if arg_params.norm:
        log_call(compute_norm, stats)(H)

    if arg_params.eigsolve:
        log_call(do_eigsolve, stats)(arg_params, H)

    if arg_params.evolve:
        log_call(do_evolve, stats)(arg_params, H, in_state, out_state)

    if arg_params.mult:
        log_call(do_mult, stats)(arg_params, H, in_state, out_state)

    if arg_params.rdm:
        keep_idxs = arg_params.keep
        if keep_idxs is None:
            keep_idxs = range(0, arg_params.L//2)
        log_call(do_rdm, stats)(in_state, keep_idxs)

    if arg_params.check_conserves:
        log_call(do_check_conserves, stats)(H)

    if arg_params.track_memory:
        # trigger memory measurement
        if H is not None:
            H.destroy_mat()
        elif in_state is not None:
            in_state.vec.destroy()

        stats['Gb_memory'] = get_max_memory_usage()

    stats['total_time'] = default_timer() - main_start

    Print('---RESULTS---')
    for k,v in stats.items():
        Print('{0}, {1:0.4f}'.format(k, v))

if __name__ == '__main__':
    main()
