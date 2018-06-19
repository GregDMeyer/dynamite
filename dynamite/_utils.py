
import numpy as np

def get_tstep(ncv,nrm,tol=1E-7):
    """
    Compute the length of a sub-step in a Expokit matrix
    exponential solve.
    """
    f = ((ncv+1)/2.72)**(ncv+1) * np.sqrt(2*np.pi*(ncv+1))
    t = ((1/nrm)*(f*tol)/(4.0*nrm))**(1/ncv)
    s = 10.0**(np.floor(np.log10(t))-1)
    return np.ceil(t/s)*s

def estimate_compute_time(t,ncv,nrm,tol=1E-7):
    """
    Estimate compute time in units of matrix multiplies, for
    an expokit exponential solve.
    """
    tstep = get_tstep(ncv,nrm,tol)
    iters = np.ceil(t/tstep)
    return ncv*iters

def popcount(x):
    '''
    Compute the number of 1 bits set in x.
    '''
    count = 0
    while x:
        count += 1
        x &= x-1
    return count

def parity(x):
    '''
    Compute the parity of x (whether the number of 1 bits set is even or odd).
    '''
    mx = np.max(x)
    i = 1
    while mx >> i:
        x = x ^ (x>>i)
        i *= 2
    return x & 1
