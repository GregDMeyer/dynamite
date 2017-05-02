
import argparse as ap
from random import uniform,seed

parser = ap.ArgumentParser(description='Benchmarking test for dynamite. Heisenberg spin chain with quenched disorder.')

parser.add_argument('-L', type=int, help='Size of the spin chain.')
parser.add_argument('-w', type=int, default=1, help='Magnitude of the disorder.')
parser.add_argument('--shell',action='store_true',help='Make a shell matrix instead of a regular matrix.')
parser.add_argument('--slepc_args',type=str,help='Arguments to pass to SLEPc.')
args = parser.parse_args()

slepc_args = args.slepc_args.split(' ')

import slepc4py
slepc4py.init(slepc_args)

from dynamite import *
from dynamite.utils import mgr
from petsc4py.PETSc import Sys
Print = Sys.Print

Print('begin building dynamite operator')

H = SumTerms(s(0)*s(1) for s in (Sigmax,Sigmay,Sigmaz))

seed(0)
for i in range(args.L):
    H += uniform(-args.w,args.w) * Sigmaz(index=i)

H.set_size(args.L)

Print('dynamite operator built. building PETSc matrix...')

m = H.get_mat()

m.view()

Print('PETSc matrix built.')
