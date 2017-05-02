
from . import initialize
initialize()

import numpy as np
from slepc4py import SLEPc
from petsc4py import PETSc

def build_state(L,init_state = 0):

    v = PETSc.Vec().create()
    v.setSizes(1<<L)
    v.setFromOptions()

    if isinstance(init_state,str):
        state_str = init_state
        init_state = 0
        if len(state_str) != L:
            raise IndexError('init_state string must have length L')
        if not all(c in ['U','D'] for c in state_str):
            raise Exception('only character U and D allowed in init_state')
        for i,c in enumerate(state_str):
            if c == 'U':
                init_state += 1<<i

    v[init_state] = 1

    v.assemblyBegin()
    v.assemblyEnd()

    return v

def vectonumpy(v):
    '''
    Collect PETSc vector v to a numpy vector on process 0.
    '''

    # collect to process 0
    sc,v0 = PETSc.Scatter.toZero(v)
    sc.begin(v,v0)
    sc.end(v,v0)

    # all processes other than 0
    if v0.getSize() == 0:
        return None

    ret = np.ndarray((v0.getSize(),),dtype=np.complex128)
    ret[:] = v0[:]

    return ret
