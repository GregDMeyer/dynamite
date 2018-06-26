
__all__ = [
    'ising'
]

from dynamite.operators import sigmax, sigmay, sigmaz, index_sum, op_sum

def ising():
    H = index_sum(op_sum(s(0)*s(1) for s in (sigmax, sigmay, sigmaz)))
    H += 0.5*index_sum(sigmaz())
    return H
