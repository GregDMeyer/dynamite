
import numpy as np
from slepc4py import init

try:
    import qutip as qtp
except ImportError:
    qtp = None

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

class _Manager:

    def __init__(self):
        self.slepc_initialized = False
        self.mfn = None

    def initialize_slepc(self,arg_list = None):
        if self.slepc_initialized:
            if arg_list is None:
                return
            else:
                raise Exception('SLEPc already initialized.')
        if arg_list is None:
            arg_list = []
        init(arg_list)
        self.slepc_initialized = True

mgr = _Manager()
# this way we can call mgr.initialize_slepc() as much as we want
# and it will only actually initialize slepc once
