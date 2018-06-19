'''
This module defines functions that operate on the MSC representation of a matrix.

The MSC_to_numpy method is the reference implementation that defines the MSC format.
'''

from itertools import chain
import numpy as np
from ._utils import parity

# TODO: move MSC_dtype here, and get int_size from backend
from .backend.backend import MSC_dtype

def MSC_to_numpy(MSC, dims, idx_to_state = None, state_to_idx = None):
    '''
    Build a NumPy array from an MSC array. This method isolates to_numpy
    from the rest of the class for testing. It also defines the MSC
    representation.

    Parameters
    ----------

    MSC : np.ndarray(dtype = MSC_dtype)
        An MSC array.

    dims : (int, int)
        The dimensions (M, N) of the matrix.

    idx_to_state : function(int), optional
        If working in a subspace, a function to map indices to states for
        the *left* subspace.

    state_to_idx : function(int), optional
        If working in a subspace, a function to map states to indices for
        the *right* subspace.

    Returns
    -------

    np.ndarray(dtype = np.complex128)
        A 2-D NumPy array which stores the matrix.
    '''

    ary = np.zeros(dims, dtype = np.complex128)

    # if these aren't supplied, they are the identity
    if idx_to_state is None:
        idx_to_state = lambda x: x

    if state_to_idx is None:
        state_to_idx = lambda x: x

    for idx in range(dims[0]):
        bra = idx_to_state(idx)
        for m,s,c in MSC:
            ket = m ^ bra
            ridx = state_to_idx(ket)
            if ridx is not None: # otherwise we went out of the subspace
                sign = 1 - 2*(parity(s & ket))
                ary[idx, ridx] += sign * c

    return ary

def MSC_sum(iterable):
    '''
    Defines the matrix addition operation for any number of MSC matrices returned by
    ``iterable``.

    Parameters
    ----------
    iterable : iter
        An iterable containing MSC representations of matrices.

    Returns
    -------
    np.ndarray
        The sum as an MSC matrix
    '''
    iterable = iter(iterable)
    # if iterable has zero items, return zero
    try:
        first = next(iterable)
    except StopIteration:
        return np.ndarray(0, dtype = MSC_dtype)

    return np.hstack(chain([first],iterable))

def MSC_product(iterable):
    '''
    Defines the matrix-matrix-matrix-... product operation for MSC matrices.

    Parameters
    ----------
    iterable : iter
        An iterable containing the MSC matrices to be multiplied together, in order.

    Returns
    -------
    np.ndarray
        The product
    '''
    vals = list(iterable)

    # an efficient way of doing the cartesian product
    all_terms = np.array(np.meshgrid(*vals)).reshape(len(vals),-1)

    # the following is the product on the MSC representation
    rtn = all_terms[0]

    # if there was a zero in the terms
    if all_terms.size == 0:
        return rtn

    for term in all_terms[1:]:
        flipped = term['masks'] & rtn['signs']
        rtn['masks'] ^= term['masks']
        rtn['signs'] ^= term['signs']
        rtn['coeffs'] *= (-1)**parity(flipped) * term['coeffs']

    return rtn

def shift(MSC, shift, wrap_idx):
    '''
    Shift an MSC representation along the spin chain. Guaranteed to not modify input,
    but not guaranteed to return a copy (could return the same object).

    Parameters
    ----------
    MSC : np.ndarray
        The input MSC representation.

    shift : int
        The number of spins to shift by.

    wrap : int or None
        The index at which to wrap around to the beginning of the spin chain.
        If None, do not wrap.

    Returns
    -------
    np.ndarray
        The shifted MSC representation.
    '''

    if shift == 0:
        return MSC

    msc = MSC.copy()

    msc['masks'] <<= shift
    msc['signs'] <<= shift

    if wrap_idx is not None:

        mask = (-1) << wrap_idx

        for v in [msc['masks'], msc['signs']]:

            # find the bits that need to wrap around
            overflow = v & mask

            # wrap them to index 0
            overflow >>= wrap_idx

            # recombine them with the ones that didn't get wrapped
            v |= overflow

            # shave off the extras that go past L
            v &= ~mask

    return msc

def combine_and_sort(MSC):
    '''
    Take an MSC representation, sort it, and combine like terms.

    Parameters
    ----------
    MSC : np.ndarray
        The input MSC representation.

    Returns
    -------
    np.ndarray
        The reduced representation (may be of a smaller dimension).
    '''

    unique, inverse = np.unique(MSC[['masks','signs']], return_inverse = True)
    rtn = np.ndarray(unique.size, dtype = MSC_dtype)

    rtn['masks'] = unique['masks']
    rtn['signs'] = unique['signs']

    rtn['coeffs'] = 0
    for i,(_,_,c) in enumerate(MSC):
        rtn[inverse[i]]['coeffs'] += c

    rtn = rtn[rtn['coeffs'] != 0]

    return rtn

def serialize(MSC):
    '''
    Take an MSC representation and spin chain length and serialize it into a
    byte string.

    The format is
    `nterms int_size masks signs coefficients`
    where `nterms`, and `int_size` are utf-8 text, including newlines, and the others
    are each just a binary blob, one after the other. `int_size` is an integer representing
    the size of the int data type used (32 or 64 bits).

    Binary values are saved in big-endian format, to be compatible with PETSc defaults.

    Parameters
    ----------
    MSC : np.array
        The MSC representation

    Returns
    -------
    bytes
        A byte string containing the serialized operator.
    '''

    rtn = b''

    rtn += (str(MSC.size)+'\n').encode('utf-8')
    rtn += (str(MSC.dtype['masks'].itemsize*8)+'\n').encode('utf-8')

    int_t = MSC.dtype[0].newbyteorder('B')
    cplx_t = np.dtype(np.complex128).newbyteorder('B')
    rtn += MSC['masks'].astype(int_t, casting='equiv', copy=False).tobytes()
    rtn += MSC['signs'].astype(int_t, casting='equiv', copy=False).tobytes()
    rtn += MSC['coeffs'].astype(cplx_t, casting='equiv', copy=False).tobytes()

    return rtn

def deserialize(data):
    '''
    Reverse the _serialize operation.

    Parameters
    ----------
    data : bytes
        The byte string containing the serialized data.

    Returns
    -------
    tuple(int, np.ndarray)
        A tuple of the form (L, MSC)
    '''

    start = 0
    stop = data.find(b'\n')
    MSC_size = int(data[start:stop])

    start = stop + 1
    stop = data.find(b'\n', start)
    int_size = int(data[start:stop])
    if int_size == 32:
        int_t = np.int32
    elif int_size == 64:
        int_t = np.int64
    else:
        raise ValueError('Invalid int_size. Perhaps file is corrupt.')

    dt = np.dtype([
        ('masks', int_t),
        ('signs', int_t),
        ('coeffs', np.complex128)
    ])
    MSC = np.ndarray(MSC_size, dtype = dt)

    # TODO: can I do this without making a copy in the calls to np.frombuffer?
    mv = memoryview(data)
    start = stop + 1
    int_MSC_bytes = MSC_size * int_size // 8

    MSC['masks'] = np.frombuffer(mv[start:start+int_MSC_bytes],
                                 dtype = np.dtype(int_t).newbyteorder('B'))
    start += int_MSC_bytes
    MSC['signs'] = np.frombuffer(mv[start:start+int_MSC_bytes],
                                 dtype = np.dtype(int_t).newbyteorder('B'))
    start += int_MSC_bytes
    MSC['coeffs'] = np.frombuffer(mv[start:],
                                  dtype = np.dtype(np.complex128).newbyteorder('B'))

    return MSC

def max_spin_idx(MSC):
    '''
    Compute the largest spin index on which the operator represented by MSC
    has support. Returns -1 for an empty operator.

    Parameters
    ----------
    MSC : np.array
        The MSC operator

    Returns
    -------
    int
        The index
    '''
    if MSC.size == 0:
        return -1

    max_op = np.max([np.max(MSC['masks']), np.max(MSC['signs'])])
    count = 0
    while max_op:
        count += 1
        max_op >>= 1
    return count-1

def nnz(MSC):
    '''
    Compute the number of nonzero elements per row of the sparse matrix representation
    of this MSC operator.
    '''
    return len(np.unique(MSC['masks']))
