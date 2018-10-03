
import numpy as np

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

    if mx == 0:
        return x

    i = 1
    # this log is necessary due to a compiler bug where 1>>64 == 1
    mx_log = np.log2(mx)
    while i <= mx_log:
        x = x ^ (x>>i)
        i *= 2
    return x & 1

def intlog2(x):
    '''
    Compute floor(log2(x)) for integer x.
    Also, intlog2(0) = -1.
    '''
    x = np.array(x)
    count = np.zeros(x.shape, dtype = np.int)
    count -= 1
    mx = np.max(x)
    while mx:
        count[x != 0] += 1
        x >>= 1
        mx >>= 1
    return count
