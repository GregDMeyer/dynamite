
import numpy as np

from .backend.backend import MSC_dtype,product_of_terms

try:
    import qutip as qtp
except ImportError:
    qtp = None

def coeff_to_str(x,signs='+-'):
    if x == 1:
        return '+' if '+' in signs else ''
    elif x == -1:
        return '-' if '-' in signs else ''
    else:
        rtn = '{:0.5g}'.format(x)
        if '+' in signs and x >= 0:
            rtn = '+' + rtn
        elif '-' not in signs and x < 0:
            rtn = rtn[1:]
        return rtn

def condense_terms(all_terms):

    all_terms.sort(order=['masks','signs'])

    combined = np.ndarray((len(all_terms),),dtype=MSC_dtype)

    i = 0
    n = 0
    maxn = len(all_terms)
    while n < maxn:
        t = all_terms[n]
        combined[i] = t
        n += 1
        while n < maxn and all_terms[n]['masks'] == t['masks'] and all_terms[n]['signs'] == t['signs']:
            combined[i]['coeffs'] += all_terms[n]['coeffs']
            n += 1
        i += 1

    combined.resize((i,)) # get rid of the memory we don't need

    return combined

def qtp_identity_product(op, index, L):

    if qtp is None:
        raise ImportError('Could not import qutip.')

    ret = None
    for i in range(L):
        if i == index:
            this_op = op
        else:
            this_op = qtp.identity(2)
        if ret is None:
            ret = this_op
        else:
            ret = qtp.tensor(this_op,ret)
    return ret

def MSC_matrix_product(terms):

    arrays = np.vstack(terms)

    sizes = np.array([a.shape[0] for a in arrays])
    all_terms = np.ndarray((np.prod(sizes),),dtype=MSC_dtype)

    prod_idxs = _pi.idxs(sizes) # see class _PermIndices below

    aT = arrays.T

    for n,idxs in enumerate(prod_idxs):
        t = np.choose(idxs,aT)
        all_terms[n] = product_of_terms(t)

    return all_terms

class _PermIndices:
    """
    store result of np.indices so as to keep that function from
    being called a zillion times
    """

    def __init__(self):
        self.p = {}

    def idxs(self,sizes):
        st = tuple(sizes)
        if st not in self.p:
            self.p[st] = np.indices(sizes).reshape((sizes.size,np.prod(sizes))).T
        return self.p[st]

_pi = _PermIndices()
