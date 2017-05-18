
import argparse as ap
from random import uniform,seed
from timeit import default_timer
from itertools import combinations

parser = ap.ArgumentParser(description='Benchmarking test for dynamite.')

parser.add_argument('-L', type=int, help='Size of the spin chain.', required=True)

parser.add_argument('-H', choices=['MBL','long_range','SYK'], default='MBL', help='Hamiltonian to use', required=True)

parser.add_argument('-w', type=int, default=1, help='Magnitude of the disorder for MBL Hamiltonian.')

parser.add_argument('--shell',action='store_true',help='Make a shell matrix instead of a regular matrix.')
parser.add_argument('--slepc_args',type=str,default='',help='Arguments to pass to SLEPc.')

parser.add_argument('--evolve',action='store_true',help='Request that the Hamiltonian evolves a state.')
parser.add_argument('--init_state',type=int,default=0,help='The initial state for the evolution.')
parser.add_argument('-t',type=float,default=1.0,help='The time to evolve for.')

parser.add_argument('--eigsolve',action='store_true',help='Request to solve for eigenvalues of the Hamiltonian.')
parser.add_argument('-nev',type=int,default=1,help='The number of eigenpairs to solve for.')
parser.add_argument('-target',type=int,help='The target for a shift-invert eigensolve.')

args = parser.parse_args()

slepc_args = args.slepc_args.split(' ')

from dynamite import initialize
initialize(slepc_args)

from dynamite.operators import Sum,Product,IndexSum,Sigmax,Sigmay,Sigmaz
from dynamite.tools import build_state,track_memory,get_max_memory_usage,get_cur_memory_usage
from dynamite.extras import Majorana as X
from petsc4py.PETSc import Sys
Print = Sys.Print

stats = {
    'MSC_build_time':None,
    'mat_build_time':None,
    'evolve_time':None,
    'eigsolve_time':None,
    'MaxRSS':None,
}

track_memory()
mem_type = 'all'

Print('begin building dynamite operator')

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

H.L = args.L
H.use_shell = args.shell

start = default_timer()

Print('nnz:',H.nnz,'\ndensity:',H.density,'\nMSC size:',H.MSC_size)
Print('dynamite operator built. building PETSc matrix...')

stats['MSC_build_time'] = default_timer() - start

start = default_timer()
H.build_mat()
stats['mat_build_time'] = default_timer() - start

Print('PETSc matrix built.')

if args.eigsolve:
    Print('beginning eigsolve...')
    start = default_timer()
    H.eigsolve(nev=args.nev,target=args.target)
    stats['eigsolve_time'] = default_timer() - start
    Print('eigsolve complete.')

if args.evolve:
    Print('beginning evolution...')
    start = default_timer()
    s = build_state(args.L,init_state=args.init_state)
    H.evolve(s,t=args.t)
    stats['evolve_time'] = default_timer() - start
    Print('evolution complete.')

H.destroy_mat()

# crap. this isn't implemented in petsc4py. I can add it to dynamite's API though...
stats['MaxRSS'] = get_max_memory_usage(mem_type)

Print('Results:')
for k,v in stats.items():
    Print('\t',k+':',v)
