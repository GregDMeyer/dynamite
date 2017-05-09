
import argparse as ap
from random import uniform,seed
from timeit import default_timer

parser = ap.ArgumentParser(description='Benchmarking test for dynamite. Heisenberg spin chain with quenched disorder.')

parser.add_argument('-L', type=int, help='Size of the spin chain.')
parser.add_argument('-w', type=int, default=1, help='Magnitude of the disorder.')
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

import slepc4py
slepc4py.init(slepc_args)

from dynamite.operators import *
from dynamite.tools import build_state
from petsc4py.PETSc import Sys
Print = Sys.Print

stats = {
    'build_time':None,
    'evolve_time':None,
    'eigsolve_time':None,
    'MaxRSS':None,
}

Print('begin building dynamite operator')

H = SumTerms(s(0)*s(1) for s in (Sigmax,Sigmay,Sigmaz))

seed(0)
for i in range(args.L):
    H += uniform(-args.w,args.w) * Sigmaz(index=i)

H.set_size(args.L)

Print('dynamite operator built. building PETSc matrix...')

start = default_timer()
H.build_mat(shell=args.shell)
stats['build_time'] = default_timer() - start

Print('PETSc matrix built.')

if args.eigsolve:
    start = default_timer()
    H.eigsolve(nev=args.nev,target=args.target)
    stats['eigsolve_time'] = default_timer() - start

if args.evolve:
    start = default_timer()
    s = build_state(args.L,init_state=args.init_state)
    H.evolve(s,t=args.t)
    stats['evolve_time'] = default_timer() - start

# crap. this isn't implemented in petsc4py. I can add it to dynamite's API though...
stats['MaxRSS'] = None

Print('Results:')
for k,v in stats.items():
    Print('\t',k+':',v)
