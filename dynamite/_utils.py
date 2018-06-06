
import numpy as np

from .backend.backend import MSC_dtype,product_of_terms

def coeff_to_str(x,signs='+-'):
    if x == 1:
        return '+' if '+' in signs else ''
    elif x == -1:
        return '-' if '-' in signs else ''
    else:
        if x == 0:
            return '0'
        elif x.imag == 0:
            rtn = '{:0.3g}'.format(x.real)
            if '+' in signs and x >= 0:
                rtn = '+' + rtn
            elif '-' not in signs and x < 0:
                rtn = rtn[1:]
            return rtn
        elif x.real == 0:
            return coeff_to_str(x.imag,signs=signs)+'i'
        else:
            rtn = '{:0.3g}'.format(x.real) + coeff_to_str(x.imag,signs='+-')
            return '(' + rtn + ')'

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

    import qutip as qtp

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

    arrays = list(terms)

    sizes = np.array([a.shape[0] for a in arrays])
    all_terms = np.ndarray((np.prod(sizes),),dtype=MSC_dtype)

    prod_idxs = _pi.idxs(sizes) # see class _PermIndices below

    t = np.ndarray((len(arrays),),dtype=MSC_dtype)
    for n,idxs in enumerate(prod_idxs):
        for i,a in enumerate(arrays):
            t[i] = a[idxs[i]]
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
