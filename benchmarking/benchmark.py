
import argparse as ap
from random import uniform,seed
from timeit import default_timer
from itertools import combinations

parser = ap.ArgumentParser(description='Benchmarking test for dynamite.')

parser.add_argument('-L', type=int, help='Size of the spin chain.', required=True)

parser.add_argument('-H', choices=['MBL','long_range','SYK','ising','XX','Load'],
                    default='MBL', help='Hamiltonian to use', required=True)
parser.add_argument('--file', help='The file from which to load the operator.')

parser.add_argument('-w', type=int, default=1, help='Magnitude of the disorder for MBL Hamiltonian.')

parser.add_argument('--shell',action='store_true',help='Make a shell matrix instead of a regular matrix.')
parser.add_argument('--slepc_args',type=str,default='',help='Arguments to pass to SLEPc.')

parser.add_argument('--evolve',action='store_true',help='Request that the Hamiltonian evolves a state.')
parser.add_argument('--init_state',type=int,default=0,help='The initial state for the evolution.')
parser.add_argument('-t',type=float,default=1.0,help='The time to evolve for.')

parser.add_argument('--mult',action='store_true',help='Simply multiply the Hamiltonian by a vector.')
parser.add_argument('--mult_count',type=int,default=1,help='Number of times to repeat the multiplication.')

parser.add_argument('--norm',action='store_true',help='Compute the norm of the matrix.')

parser.add_argument('--eigsolve',action='store_true',help='Request to solve for eigenvalues of the Hamiltonian.')
parser.add_argument('--nev',type=int,default=1,help='The number of eigenpairs to solve for.')
parser.add_argument('--target',type=int,help='The target for a shift-invert eigensolve.')

args = parser.parse_args()

slepc_args = args.slepc_args.split(' ')

from dynamite import config
config.initialize(slepc_args)

from dynamite.operators import Sum,Product,IndexSum,Sigmax,Sigmay,Sigmaz,Load
from dynamite.tools import build_state,track_memory,get_max_memory_usage
from dynamite.extras import Majorana as X
from petsc4py.PETSc import Sys,NormType
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

config.global_L = args.L

if __debug__:
    Print('begin building dynamite operator')
else:
    Print('---ARGUMENTS---')
    for k,v in vars(args).items():
        Print(str(k)+','+str(v))

if args.H == 'MBL':
    # dipolar interaction
    H = IndexSum(Sum(s(0)*s(1) for s in (Sigmax,Sigmay,Sigmaz)))
    # quenched disorder in z direction
    seed(0)
    for i in range(args.L):
        H += uniform(-args.w,args.w) * Sigmaz(index=i)
elif args.H == 'long_range':
    # long-range ZZ interaction
    H = Sum(IndexSum(Sigmaz(0)*Sigmaz(i)) for i in range(1,args.L))
    # nearest neighbor XX
    H += 0.5 * IndexSum(Sigmax(0)*Sigmax(1))
    # some other fields
    H += Sum(0.1*IndexSum(s()) for s in [Sigmax,Sigmay,Sigmaz])
elif args.H == 'SYK':
    seed(0)
    H = Sum(uniform(-1,1)*Product(X(idx) for idx in idxs) for idxs in combinations(range(args.L*2),4))
elif args.H == 'ising':
    H = IndexSum(Sigmaz(0)*Sigmaz(1)) + 0.2*IndexSum(Sigmax())
elif args.H == 'XX':
    H = IndexSum(Sigmax(0)*Sigmax(1))
elif args.H == 'Load':
    H = Load(args.file)
else:
    raise ValueError('Unrecognized Hamiltonian.')

start = default_timer()

if __debug__:
    Print('nnz:',H.nnz,'\ndensity:',H.density,'\nMSC size:',H.MSC_size)
    Print('dynamite operator built. building PETSc matrix...')

stats['MSC_build_time'] = default_timer() - start

H.use_shell = args.shell

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
    s = build_state(state=args.init_state)
    H.evolve(s,t=args.t)
    stats['evolve_time'] = default_timer() - start
    if __debug__:
        Print('evolution complete.')

if args.mult:
    if __debug__:
        Print('beginning multiplication...')
    start = default_timer()
    s = build_state()
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
