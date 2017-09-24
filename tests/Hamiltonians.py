
import dynamite.operators as do
import numpy as np
from numpy_operators import *

# TODO: use numpy sparse here
__all__ = ['XXYY',
           'XXYYZZ',
           'ising',
           'longrange']

def XXYY(L):
    d = do.IndexSum(do.Sum(s(0)*s(1) for s in [do.Sigmax,do.Sigmay]))
    d.L = L

    n = np.zeros((2**L,2**L),dtype=np.complex128)
    for s in 'xy':
        n += sum(np_sigmai(s,i,L).dot(np_sigmai(s,i+1,L)) for i in range(L-1))

    return d,n

def XXYYZZ(L):
    d = do.IndexSum(do.Sum(s(0)*s(1) for s in [do.Sigmax,do.Sigmay,do.Sigmaz]))
    d.L = L

    n = np.zeros((2**L,2**L),dtype=np.complex128)
    for s in 'xyz':
        n += sum(np_sigmai(s,i,L).dot(np_sigmai(s,i+1,L)) for i in range(L-1))

    return d,n

def ising(L):
    d = do.IndexSum(do.Sigmaz(0)*do.Sigmaz(1)) + 0.2*do.IndexSum(do.Sigmax())
    d.L = L

    n = np.zeros((2**L,2**L),dtype=np.complex128)
    n += sum(np_sigmaz(i,L).dot(np_sigmaz(i+1,L)) for i in range(L-1))
    n += 0.2*sum(np_sigmax(i,L) for i in range(L))

    return d,n

def longrange(L):
    d = do.Sum(do.IndexSum(do.Sigmaz(0)*do.Sigmaz(i)) for i in range(1,L-1))
    d += 0.5*do.IndexSum(do.Sigmax(0)*do.Sigmax(1))
    d += 0.1*do.Sum(do.IndexSum(s()) for s in (do.Sigmax,do.Sigmay,do.Sigmaz))
    d.L = L

    n = np.zeros((2**L,2**L),dtype=np.complex128)

    for j in range(1,L-1):
        n += sum(np_sigmaz(i,L).dot(np_sigmaz(i+j,L)) for i in range(L-j))

    n += 0.5*sum(np_sigmax(i,L).dot(np_sigmax(i+1,L)) for i in range(L-1))
    n += 0.1*sum(np_sigmai(t,i,L) for i in range(L) for t in 'xyz')

    return d,n
