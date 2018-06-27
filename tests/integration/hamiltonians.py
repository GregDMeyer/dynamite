
__all__ = [
    'ising',
    'long_range'
]

import numpy as np
from dynamite.operators import sigmax, sigmay, sigmaz, index_sum, op_sum

def ising(L = None):
    '''
    Classic quantum Ising model with a transverse field.
    '''
    H = index_sum(sigmaz(0)*sigmaz(1), size = L)
    H += 0.5*index_sum(sigmax())
    return H

def long_range(L = None):
    '''
    Long-range (polynomially decaying) interaction with closed boundary conditions.
    '''
    # decay exponent
    alpha = 1.13

    # nearest neighbor XX
    H = index_sum(sigmax(0)*sigmax(1), size = L, boundary = 'closed')

    # long range ZZ interaction
    H += op_sum(index_sum(1/(i**alpha)*sigmaz(0)*sigmaz(i)) for i in range(1, H.get_length()))

    # some uniform fields
    H += index_sum(0.5*sigmax())
    H += index_sum(0.3*sigmay())
    H += index_sum(0.1*sigmaz())

    return H

def localized(L = None):
    '''
    Random-field Heisenberg.
    '''
    H = index_sum(op_sum(s(0)*s(1) for s in (sigmax, sigmay, sigmaz)), size = L)
    H += op_sum(np.random.uniform(-1, 1)*sigmaz(i) for i in range(H.get_length()))
    return H
