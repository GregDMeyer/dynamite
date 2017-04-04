
from slepc4py import init

from slepc4py import SLEPc
from petsc4py import PETSc

import atexit

def build_state(L,init_state = 0):
    mgr.initialize_slepc()

    v = PETSc.Vec().create()
    v.setSizes(1<<L)
    v.setFromOptions()

    v[init_state] = 1

    v.assemblyBegin()
    v.assemblyEnd()

    return v

def evolve(x,H=None,t=None,result=None,tol=None,mfn=None):

    mgr.initialize_slepc()

    if result is None:
        result = H.get_mat().createVecs(side='l')

    if mfn is None:
        mfn = SLEPc.MFN().create()
        mfn.setType('expokit')

        f = mfn.getFN()
        f.setType(SLEPc.FN.Type.EXP)

        mfn.setFromOptions()

        if t is None or H is None:
            raise Exception('Must supply t and H if not supplying mfn to evolve')

    if tol is not None:
        mfn.setTolerances(tol)

    if H is not None:
        mfn.setOperator(H.get_mat())

    if t is not None:
        f = mfn.getFN()
        f.setScale(-1j*t)

    mfn.solve(x,result)

    return result

class _Manager:

    def __init__(self,printer):
        self.slepc_initialized = False
        self.mfn = None
        self.printer = printer

    def initialize_slepc(self,arg_list = None):
        if self.slepc_initialized:
            if arg_list is None:
                return
            else:
                raise Exception('SLEPc already initialized.')
        if arg_list is None:
            arg_list = []
        init(arg_list)
        self.print = PETSc.Sys.Print
        self.slepc_initialized = True
        self.printer.set_print_fn(PETSc.Sys.Print)

# so that we can handle multiprocess printing
class _Printer:

    def __init__(self):
        self.print_fn = print

    def __call__(self,*args,**kwargs):
        self.print_fn(*args,**kwargs)

    def set_print_fn(self,fn):
        self.print_fn = fn

Print = _Printer()
mgr = _Manager(Print)
# this way we can call initialize.slepc() as much as we want
# and it will only actually initialize slepc once


