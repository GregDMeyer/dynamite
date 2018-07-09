'''
This module defines functions that operate on the MSC representation of a matrix.

The MSC_to_numpy method is the reference implementation that defines the MSC format.
'''

from itertools import chain
import numpy as np
import scipy.sparse
from .bitwise import parity, intlog2

from ._backend.bbuild import dnm_int_t

msc_dtype = np.dtype([('masks', dnm_int_t),
                      ('signs', dnm_int_t),
                      ('coeffs', np.complex128)])

def msc_to_numpy(msc, dims, idx_to_state = None, state_to_idx = None, sparse = True):
    '''
    Build a NumPy array from an MSC array. This method defines the MSC
    representation.

    Parameters
    ----------

    MSC : np.ndarray(dtype = msc_dtype)
        An MSC array.

    dims : (int, int)
        The dimensions (M, N) of the matrix.

    idx_to_state : function(int), optional
        If working in a subspace, a function to map indices to states for
        the *left* subspace.

    state_to_idx : function(int), optional
        If working in a subspace, a function to map states to indices for
        the *right* subspace.

    sparse : bool, optional
        Whether to return a scipy sparse matrix or a dense numpy array.

    Returns
    -------
    scipy.spmatrix or np.ndarray (dtype = np.complex128)
        The matrix
    '''
    msc = np.array(msc, copy = False, dtype = msc_dtype)
    data = np.ndarray(msc.size * np.min(dims), dtype = np.complex128)
    # data[:] = -1 # for testing if we have correctly sized buffers
    row_idxs = np.ndarray(data.size, dtype = dnm_int_t)
    col_idxs = np.ndarray(data.size, dtype = dnm_int_t)
    mat_idx = 0

    # if these aren't supplied, they are the identity
    if idx_to_state is None:
        idx_to_state = lambda x: x

    if state_to_idx is None:
        state_to_idx = lambda x: x

    for idx in range(dims[0]):
        bra = idx_to_state(idx)
        ket = msc['masks'] ^ bra
        ridx = state_to_idx(ket)
        # TODO: do we need to be careful about unsigned integers here?

        good = np.nonzero(ridx != -1)[0]
        nnew = len(good)
        if nnew == 0:
            continue

        good_ridx = ridx[good]
        good_kets = ket[good]
        sign = 1 - 2*(parity(msc['signs'][good] & good_kets))

        nnew = len(good)
        data[mat_idx:mat_idx+nnew] = sign * msc['coeffs'][good]
        row_idxs[mat_idx:mat_idx+nnew] = idx
        col_idxs[mat_idx:mat_idx+nnew] = good_ridx
        mat_idx += nnew

    # trim to the amount we used
    data = data[:mat_idx]
    row_idxs = row_idxs[:mat_idx]
    col_idxs = col_idxs[:mat_idx]

    ary = scipy.sparse.csc_matrix((data, (row_idxs, col_idxs)), shape = dims)

    if not sparse:
        ary = ary.toarray()

    return ary

def msc_sum(iterable):
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
        return np.ndarray(0, dtype = msc_dtype)

    return np.hstack(chain([first],iterable))

def msc_product(iterable):
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

def shift(msc, shift_idx, wrap_idx):
    '''
    Shift an MSC representation along the spin chain. Guaranteed to not modify input,
    but not guaranteed to return a copy (could return the same object).

    Parameters
    ----------
    MSC : np.ndarray
        The input MSC representation.

    shift_idx : int
        The number of spins to shift by.

    wrap : int or None
        The index at which to wrap around to the beginning of the spin chain.
        If None, do not wrap.

    Returns
    -------
    np.ndarray
        The shifted MSC representation.
    '''

    if shift_idx == 0:
        return msc

    msc = msc.copy()

    msc['masks'] <<= shift_idx
    msc['signs'] <<= shift_idx

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

def combine_and_sort(msc):
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

    unique, inverse = np.unique(msc[['masks','signs']], return_inverse = True)
    rtn = np.ndarray(unique.size, dtype = msc.dtype)

    rtn['masks'] = unique['masks']
    rtn['signs'] = unique['signs']

    rtn['coeffs'] = 0
    for i,(_,_,c) in enumerate(msc):
        rtn[inverse[i]]['coeffs'] += c

    rtn = rtn[rtn['coeffs'] != 0]

    return rtn

def serialize(msc):
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

    rtn += (str(msc.size)+'\n').encode('utf-8')
    rtn += (str(msc.dtype['masks'].itemsize*8)+'\n').encode('utf-8')

    int_t = msc.dtype[0].newbyteorder('B')
    cplx_t = np.dtype(np.complex128).newbyteorder('B')
    rtn += msc['masks'].astype(int_t, casting='equiv', copy=False).tobytes()
    rtn += msc['signs'].astype(int_t, casting='equiv', copy=False).tobytes()
    rtn += msc['coeffs'].astype(cplx_t, casting='equiv', copy=False).tobytes()

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
    msc_size = int(data[start:stop])

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
    msc = np.ndarray(msc_size, dtype = dt)

    # TODO: can I do this without making a copy in the calls to np.frombuffer?
    mv = memoryview(data)
    start = stop + 1
    int_msc_bytes = msc_size * int_size // 8

    msc['masks'] = np.frombuffer(mv[start:start+int_msc_bytes],
                                 dtype = np.dtype(int_t).newbyteorder('B'))
    start += int_msc_bytes
    msc['signs'] = np.frombuffer(mv[start:start+int_msc_bytes],
                                 dtype = np.dtype(int_t).newbyteorder('B'))
    start += int_msc_bytes
    msc['coeffs'] = np.frombuffer(mv[start:],
                                  dtype = np.dtype(np.complex128).newbyteorder('B'))

    return msc

def max_spin_idx(msc):
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
    if msc.size == 0:
        return -1

    max_op = np.max([np.max(msc['masks']), np.max(msc['signs'])])
    return intlog2(max_op)

def nnz(msc):
    '''
    Compute the number of nonzero elements per row of the sparse matrix representation
    of this MSC operator.
    '''
    return len(np.unique(msc['masks']))

def table(msc, L):
    '''
    Build a table in string format that shows all of the terms in the MSC matrix.

    Displays only the real part of coefficients, since complex coefficients would imply
    non-Hermitian matrices.
    '''

    rtn = '   coeff. | {pad}operator{pad} \n' +\
          '====================={epad}\n'

    npad = max(L - 8, 0)
    rtn = rtn.format(pad = ' '*(npad//2), epad = '='*npad)

    terms = []
    for m, s, c in msc:

        if 1E-2 < abs(c) < 1E2:
            term = ' {:8.3f} '
        else:
            term = ' {:.2e} '

        term += '| '

        for i in range(L):
            maskbit = (m >> i) & 1
            signbit = (s >> i) & 1

            term += [['-', 'Z'],
                     ['X', 'Y']][maskbit][signbit]

            if maskbit and signbit:
                c *= -1j

        term = term.format(c.real)
        terms.append(term)

    rtn += '\n'.join(terms)

    return rtn
