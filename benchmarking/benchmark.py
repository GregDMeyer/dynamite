
import argparse as ap
from random import uniform,seed
from timeit import default_timer
from itertools import combinations

from dynamite import config
from dynamite.states import State
from dynamite.operators import sigmax, sigmay, sigmaz
from dynamite.operators import op_sum, op_product, index_sum, load_from_file
from dynamite.tools import track_memory, get_max_memory_usage
from dynamite.extras import majorana
from dynamite.subspace import Parity

parser = ap.ArgumentParser(description='Benchmarking test for dynamite.')

parser.add_argument('-L', type=int, required=True,
                    help='Size of the spin chain.')

parser.add_argument('-H', choices=['MBL','long_range','SYK','ising','XX','Load'],
                    default='MBL', required=True, help='Hamiltonian to use')
parser.add_argument('--file', help='The file from which to load the operator.')

parser.add_argument('-w', type=int, default=1,
                    help='Magnitude of the disorder for MBL Hamiltonian.')

parser.add_argument('--shell', action='store_true',
                    help='Make a shell matrix instead of a regular matrix.')
parser.add_argument('--slepc_args', type=str, default='',
                    help='Arguments to pass to SLEPc.')
parser.add_argument('--parity', type=int, choices=[0,1], default=-1,
                    help='Work in a parity subspace.')

parser.add_argument('--evolve', action='store_true',
                    help='Request that the Hamiltonian evolves a state.')
parser.add_argument('--init_state', type=int, default=0,
                    help='The initial state for the evolution.')
parser.add_argument('-t', type=float, default=1.0,
                    help='The time to evolve for.')

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
parser.add_argument('--target', type=int,
                    help='The target for a shift-invert eigensolve.')

args = parser.parse_args()
slepc_args = args.slepc_args.split(' ')
config.initialize(slepc_args)

from petsc4py.PETSc import Sys, NormType
Print = Sys.Print

stats = {}

# stats = {
#     'MSC_build_time':None,
#     'mat_build_time':None,
#     'mult_time':None,
#     'evolve_time':None,
#     'eigsolve_time':None,
#     'MaxRSS':None,
# }

#track_memory()
mem_type = 'all'

config.L = args.L

if __debug__:
    Print('begin building dynamite operator')
else:
    Print('---ARGUMENTS---')
    for k,v in vars(args).items():
        Print(str(k)+','+str(v))

start = default_timer()

if args.H == 'MBL':
    # dipolar interaction
    H = index_sum(op_sum(s(0)*s(1) for s in (sigmax, sigmay, sigmaz)))
    # quenched disorder in z direction
    seed(0)
    for i in range(args.L):
        H += uniform(-args.w,args.w) * sigmaz(i)

elif args.H == 'long_range':
    # long-range ZZ interaction
    H = op_sum(index_sum(sigmaz(0)*sigmaz(i)) for i in range(1, args.L))
    # nearest neighbor XX
    H += 0.5 * index_sum(sigmax(0)*sigmax(1))
    # some other fields
    H += sum(0.1*index_sum(s()) for s in [sigmax, sigmay, sigmaz])

elif args.H == 'SYK':
    seed(0)
    H = sum(uniform(-1,1)*op_product(majorana(idx) for idx in idxs)
            for idxs in combinations(range(args.L*2),4))

elif args.H == 'ising':
    H = index_sum(sigmaz(0)*sigmaz(1)) + 0.2*index_sum(sigmax())

elif args.H == 'XX':
    H = index_sum(sigmax(0)*sigmax(1))

elif args.H == 'Load':
    H = load_from_file(args.file)

else:
    raise ValueError('Unrecognized Hamiltonian.')

if __debug__:
    Print('nnz:', H.nnz, '\ndensity:', H.density, '\nMSC size:', H.msc_size)
    Print('dynamite operator built. building PETSc matrix...')

stats['MSC_build_time'] = default_timer() - start

H.shell = args.shell
if args.parity != -1:
    H.subspace = Parity(space=args.parity)

start = default_timer()
H.build_mat()
stats['mat_build_time'] = default_timer() - start

if __debug__:
    Print('PETSc matrix built.')

# compute the norm
if args.norm:
    if __debug__:
        Print('computing norm...')
    start = default_timer()
    H.get_mat().norm(NormType.INFINITY)
    stats['norm_time'] = default_timer() - start
    if __debug__:
        Print('norm compute complete.')

if args.eigsolve:
    if __debug__:
        Print('beginning eigsolve...')
    start = default_timer()
    H.eigsolve(nev=args.nev,target=args.target)
    stats['eigsolve_time'] = default_timer() - start
    if __debug__:
        Print('eigsolve complete.')

if args.evolve:
    if __debug__:
        Print('beginning evolution...')
    start = default_timer()
    s = State(state=args.init_state, subspace = H.subspace)
    H.evolve(s, t=args.t)
    stats['evolve_time'] = default_timer() - start
    if __debug__:
        Print('evolution complete.')

if args.mult:
    if __debug__:
        Print('beginning multiplication...')
    start = default_timer()
    s = State(state=args.init_state, subspace = H.subspace)
    r = s.copy()
    for _ in range(args.mult_count):
        H.get_mat().mult(s,r)
        H.get_mat().mult(r,s)
    stats['mult_time'] = default_timer() - start
    if __debug__:
        Print('multiplication complete.')

H.destroy_mat()

#stats['MaxRSS'] = get_max_memory_usage(mem_type)

Print('---RESULTS---')
for k,v in stats.items():
    Print(str(k)+','+str(v))
