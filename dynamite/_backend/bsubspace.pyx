
# fix cython "can't resolve package from __spec__ or __package__" warning
from __future__ import absolute_import

import numpy as np
cimport numpy as np

from .bbuild import dnm_int_t

cdef extern from "bsubspace_impl.h":

    ctypedef int PetscInt

    ctypedef struct data_Full:
        int L

    ctypedef struct data_Parity:
        int L
        int space

    ctypedef struct data_Auto:
        int L
        int dim
        int rdim
        int* state_map
        int* state_rmap

    ctypedef enum subspace_type:
        _FULL "FULL"
        _PARITY "PARITY"
        _AUTO "AUTO"

    PetscInt Dim_Full(data_Full* data);
    void S2I_Full_array(int n, const data_Full* data, const PetscInt* states, PetscInt* idxs);
    void I2S_Full_array(int n, const data_Full* data, const PetscInt* idxs, PetscInt* states);

    PetscInt Dim_Parity(const data_Parity* data);
    void S2I_Parity_array(int n, const data_Parity* data, const PetscInt* states, PetscInt* idxs);
    void I2S_Parity_array(int n, const data_Parity* data, const PetscInt* idxs, PetscInt* states);

    PetscInt Dim_Auto(const data_Auto* data);
    void S2I_Auto_array(int n, const data_Auto* data, const PetscInt* states, PetscInt* idxs);
    void I2S_Auto_array(int n, const data_Auto* data, const PetscInt* idxs, PetscInt* states);

#####

class SubspaceType:
    FULL = _FULL
    PARITY = _PARITY
    AUTO = _AUTO

#####

cdef class CFull:
    cdef data_Full data[1]

    def __init__(self, int L):
        self.data[0].L = L

cdef class CParity:
    cdef data_Parity data[1]

    def __init__(self, int L, int space):
        self.data[0].L = L
        self.data[0].space = space

# need to be careful here that the numpy arrays don't get freed
# if we will use this class in shell matrices, should copy maps
cdef class CAuto:
    cdef data_Auto data[1]

    def __init__(self, PetscInt L, PetscInt [:] state_map, PetscInt [:] state_rmap):
        self.data[0].L = L
        self.data[0].dim = state_map.size
        self.data[0].rdim = state_rmap.size
        self.data[0].state_map = &state_map[0]
        self.data[0].state_rmap = &state_rmap[0]

#####

cdef void set_data_pointer(int sub_type, object data, void** ptr):
    if sub_type == _FULL:
        set_data_pointer_Full(data, ptr)
    elif sub_type == _PARITY:
        set_data_pointer_Parity(data, ptr)
    elif sub_type == _AUTO:
        set_data_pointer_Auto(data, ptr)
    else:
        raise ValueError('Invalid data type %s' % str(type(data)))

cdef void set_data_pointer_Full(CFull data, void** ptr):
    ptr[0] = data.data

cdef void set_data_pointer_Parity(CParity data, void** ptr):
    ptr[0] = data.data

cdef void set_data_pointer_Auto(CAuto data, void** ptr):
    ptr[0] = data.data

#####

def get_dimension_Full(CFull data):
    return Dim_Full(data.data)

def get_dimension_Parity(CParity data):
    return Dim_Parity(data.data)

def get_dimension_Auto(CAuto data):
    return Dim_Auto(data.data)

#####

def idx_to_state_Full(PetscInt [:] idxs, CFull data):
    states_np = np.ndarray(idxs.size, dtype = dnm_int_t)
    cdef PetscInt [:] states = states_np
    I2S_Full_array(idxs.size, data.data, &idxs[0], &states[0])
    return states_np

def idx_to_state_Parity(PetscInt [:] idxs, CParity data):
    states_np = np.ndarray(idxs.size, dtype = dnm_int_t)
    cdef PetscInt [:] states = states_np
    I2S_Parity_array(idxs.size, data.data, &idxs[0], &states[0])
    return states_np

def idx_to_state_Auto(PetscInt [:] idxs, CAuto data):
    states_np = np.ndarray(idxs.size, dtype = dnm_int_t)
    cdef PetscInt [:] states = states_np
    I2S_Auto_array(idxs.size, data.data, &idxs[0], &states[0])
    return states_np

#####

def state_to_idx_Full(PetscInt [:] states, CFull data):
    idxs_np = np.ndarray(states.size, dtype = dnm_int_t)
    cdef PetscInt [:] idxs = idxs_np
    S2I_Full_array(states.size, data.data, &states[0], &idxs[0])
    return idxs_np

def state_to_idx_Parity(PetscInt [:] states, CParity data):
    idxs_np = np.ndarray(states.size, dtype = dnm_int_t)
    cdef PetscInt [:] idxs = idxs_np
    S2I_Parity_array(states.size, data.data, &states[0], &idxs[0])
    return idxs_np

def state_to_idx_Auto(PetscInt [:] states, CAuto data):
    idxs_np = np.ndarray(states.size, dtype = dnm_int_t)
    cdef PetscInt [:] idxs = idxs_np
    S2I_Auto_array(states.size, data.data, &states[0], &idxs[0])
    return idxs_np

#####

cdef extern int __builtin_parity(unsigned int x)

def compute_rcm(PetscInt [:] masks, PetscInt [:] signs, np.complex128_t [:] coeffs,
                PetscInt [:] state_map, PetscInt [:] state_rmap, PetscInt start,
                PetscInt L):

    cdef PetscInt full_dim = 2**L
    cdef PetscInt nnz = len(np.unique(masks))
    cdef PetscInt map_idx, i, msc_idx, cur_mask, edge, sign
    cdef np.complex128_t tot_coeff

    map_idx = 0
    state_map[map_idx] = start
    state_rmap[start] = map_idx
    map_idx += 1

    for i in range(state_map.size):
        if i == map_idx:
            break

        state = state_map[i]

        cur_mask = masks[0]
        tot_coeff = 0
        for msc_idx in range(masks.size):

            sign = __builtin_parity(state & signs[msc_idx])
            tot_coeff += (1-2*sign)*coeffs[msc_idx]

            if (msc_idx+1 == masks.size or masks[msc_idx+1] != cur_mask):
                edge = state ^ cur_mask
                if state_rmap[edge] == -1 and tot_coeff != 0:
                    if map_idx >= state_map.size:
                        raise RuntimeError('state_map size too small')
                    state_map[map_idx] = edge
                    state_rmap[edge] = map_idx
                    map_idx += 1

                tot_coeff = 0

                if msc_idx+1 < masks.size:
                    cur_mask = masks[msc_idx+1]

    return map_idx
