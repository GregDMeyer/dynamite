
__all__ = [
    'ising',
    'long_range',
    'localized',
    'syk'
]

from itertools import combinations
import numpy as np
from dynamite import config
from dynamite.operators import sigmax, sigmay, sigmaz
from dynamite.operators import index_sum, op_sum, op_product
from dynamite.extras import majorana

def ising(L = None):
    '''
    Classic quantum Ising model with a transverse field.
    '''
    H = index_sum(sigmaz(0)*sigmaz(1), size = L)
    H += 0.5*index_sum(sigmax(), size = H.get_length())
    return H

def long_range(L = None):
    '''
    Long-range (polynomially decaying) interaction.
    '''
    # decay exponent
    alpha = 1.13

    # nearest neighbor XX
    H = index_sum(sigmax(0)*sigmax(1), size = L)

    # long range ZZ interaction
    H += op_sum(index_sum(1/(i**alpha)*sigmaz(0)*sigmaz(i), size = H.get_length())
                for i in range(1, H.get_length()))

    # some uniform fields
    H += index_sum(0.5*sigmax(), H.get_length())
    H += index_sum(0.3*sigmay(), H.get_length())
    H += index_sum(0.1*sigmaz(), H.get_length())

    return H

def localized(L = None):
    '''
    Random-field Heisenberg.
    '''
    np.random.seed(0)
    H = index_sum(op_sum(s(0)*s(1) for s in (sigmax, sigmay, sigmaz)), size = L)
    H += op_sum(np.random.uniform(-1, 1)*sigmaz(i) for i in range(H.get_length()))
    return H

def syk(L = None):
    '''
    Sachdev-Ye-Kitaev model.
    '''
    np.random.seed(0)

    if L is None:
        L = config.L

    # only compute the majoranas once
    majoranas = [majorana(i) for i in range(L*2)]

    H = op_sum(op_product(majoranas[idx] for idx in idxs).scale(np.random.uniform(-1,1))
               for idxs in combinations(range(L*2),4))

    return H
