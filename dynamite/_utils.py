
import numpy as np
from slepc4py import init

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
        return ('+' if '+' in signs else '')+str(x)

def term_dtype():
    return np.dtype([('masks',np.int32),('signs',np.int32),('coeffs',np.complex128)])

def product_of_terms(factors):

    prod = np.array([(0,0,1)],dtype=term_dtype())
    for factor in factors:

        # keep the sign correct after spin flips.
        # this is crucial... otherwise everything
        # would commute!
        flipped = prod['masks'] & factor['signs']
        n_flip = 0
        while flipped:
            n_flip += 1
            flipped = flipped & (flipped-1)

        prod['masks'] = prod['masks'] ^ factor['masks']
        prod['signs'] = prod['signs'] ^ factor['signs']

        prod['coeffs'] *= factor['coeffs'] * ( (-1)**n_flip )
    return prod

def condense_terms(all_terms):

    all_terms.sort(order=['masks','signs'])

    combined = np.ndarray((len(all_terms),),dtype=term_dtype())

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

