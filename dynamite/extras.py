
from .operators import Sigmax,Sigmay,Sigmaz,PiProd

def commutator(o1,o2):
    return o1*o2 - o2*o1

def Majorana(ind):

    spin_index = ind//2
    parity = ind % 2
    if parity:
        m = Sigmay(spin_index)
    else:
        m = Sigmax(spin_index)

    if spin_index > 0:
        m = PiProd( Sigmaz(),max_i=spin_index-1) * m

    return m
