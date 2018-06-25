
# fix cython "can't resolve package from __spec__ or __package__" warning
from __future__ import absolute_import

import numpy as np
cimport numpy as np

from .bbuild import dnm_int_t

cdef extern from "bsubspace_impl.h":

    ctypedef int dnm_cint_t

    ctypedef enum subspace_type:
        _FULL "FULL"
        _PARITY "PARITY"
        _AUTO "AUTO"

    void Parity_I2S(dnm_cint_t* idxs, int n, int L, int space, dnm_cint_t* states)
    void Parity_S2I(dnm_cint_t* states, int n, int L, int space, dnm_cint_t* idxs)

    void SzTot_I2S(dnm_cint_t* idxs, int n, int L, int space, dnm_cint_t* states)
    void SzTot_S2I(dnm_cint_t* states, int n, int L, int space, dnm_cint_t* idxs)

class SubspaceType:
    FULL = _FULL
    PARITY = _PARITY
    AUTO = _AUTO

def parity_i2s(dnm_cint_t [:] idxs, int L, int space):
    states_np = np.ndarray(idxs.size, dtype = dnm_int_t)
    cdef dnm_cint_t [:] states = states_np

    Parity_I2S(&idxs[0], idxs.size, L, space, &states[0])

    return states_np

def parity_s2i(dnm_cint_t [:] states, int L, int space):
    idxs_np = np.ndarray(states.size, dtype = dnm_int_t)
    cdef dnm_cint_t [:] idxs = idxs_np

    Parity_S2I(&states[0], idxs.size, L, space, &idxs[0])

    return idxs_np
