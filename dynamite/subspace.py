
import numpy as np
from copy import deepcopy

from . import validate
from ._backend import bsubspace
from .bitwise import parity, intlog2

# TODO: allow automatically choosing subspace for operators

def class_to_enum(subspace_type):
    '''
    Convert the class types used in the Python frontend to the enum values
    used in the C backend.
    '''
    to_enum = {
        Full   : bsubspace.SubspaceType.FULL,
        Parity : bsubspace.SubspaceType.PARITY,
        Auto   : bsubspace.SubspaceType.AUTO,
    }
    return to_enum[subspace_type]

class Subspace:

    def __init__(self, space):
        self._L = None
        self._space = self._check_space(space)

    def __eq__(self,s):
        if not isinstance(s,Subspace):
            raise ValueError('Cannot compare Subspace to non-Subspace type')
        return type(s) == type(self) and s.L == self.L and s.space == self.space

    @classmethod
    def _check_space(cls, value):
        raise NotImplementedError()

    @property
    def space(self):
        """
        A parameter defining the subspace---for example, for parity,
        it might be even or odd.
        """
        return self._space

    @property
    def L(self):
        '''
        The spin chain length corresponding to this space.
        '''
        return self._L

    @classmethod
    def _check_L(cls, L, space):
        raise NotImplementedError()

    @L.setter
    def L(self, value):
        # check that this value of L is compatible with the subspace
        value = validate.L(value)
        value = self._check_L(value, self.space)
        self._L = value

    @classmethod
    def _get_dimension(cls, L, space):
        raise NotImplementedError()

    def get_dimension(self):
        """
        Get the dimension of the subspace.
        """
        return self._get_dimension(self.L, self.space)

    @classmethod
    def _idx_to_state(cls, idx, L, space):
        raise NotImplementedError

    def idx_to_state(self, idx):
        """
        Maps an index to an integer that in binary corresponds to the spin configuration.
        Vectorized implementation allows passing a numpy array of indices as idx.
        """
        if self.L is None:
            raise ValueError('Must set spin chain size for Subspace before calling '
                             'idx_to_state.')
        idx = np.array(idx, copy = False, dtype = bsubspace.dnm_int_t).reshape((-1,))
        return self._idx_to_state(idx, self.L, self.space)

    @classmethod
    def _state_to_idx(cls, state, L, space):
        raise NotImplementedError

    def state_to_idx(self, state):
        """
        The inverse mapping of :meth:`idx_to_state`.
        """
        if self.L is None:
            raise ValueError('Must set spin chain size for Subspace before calling '
                             'state_to_idx.')
        state = np.array(state, copy = False, dtype = bsubspace.dnm_int_t).reshape((-1,))
        return self._state_to_idx(state, self.L, self.space)

    def copy(self):
        return deepcopy(self)

class Full(Subspace):

    def __init__(self, space = None):
        Subspace.__init__(self, space = space)

    @classmethod
    def _get_dimension(cls, L, space):
        return 1 << L

    @classmethod
    def _check_space(cls, value):
        if not (value is None or value == 0):
            raise ValueError('Only valid choice for full space is "None" or 0.')
        return 0

    @classmethod
    def _check_L(cls, L, space):
        # any L that passes our normal validation checks works
        return L

    @classmethod
    def _idx_to_state(cls, idx, L, space):
        dim = cls._get_dimension(L, space)
        out_of_range = np.logical_or(idx < 0, idx >= dim)
        idx[out_of_range] = -1
        return idx

    @classmethod
    def _state_to_idx(cls, state, L, space):
        dim = cls._get_dimension(L, space)
        out_of_range = np.logical_or(state < 0, state >= dim)
        state[out_of_range] = -1
        return state

class Parity(Subspace):
    '''
    The subspaces of states in which the number of up spins is even or odd.

    Parameters
    ----------
    space : int
        Either 0 or 'even' for the even subspace, or 1 or 'odd' for the other.
    '''

    @classmethod
    def _get_dimension(cls, L, space):
        return 1 << (L-1)

    @classmethod
    def _check_space(cls, value):
        if value in [0,'even']:
            return 0
        elif value in [1,'odd']:
            return 1
        else:
            raise ValueError('Invalid parity space "'+str(value)+'" '
                             '(valid choices are 0, 1, "even", or "odd")')

    @classmethod
    def _check_L(cls, L, space):
        # any L that passes our normal validation checks works
        return L

    @classmethod
    def _idx_to_state(cls, idx, L, space):
        dim = cls._get_dimension(L, space)
        out_of_range = np.logical_or(idx < 0, idx >= dim)
        state = bsubspace.parity_i2s(idx, L, space)
        state[out_of_range] = -1
        return state

    @classmethod
    def _state_to_idx(cls, state, L, space):
        par = parity(state)
        max_spin_idx = intlog2(state)
        invalid = np.logical_or(par != space, max_spin_idx >= L)
        idxs = bsubspace.parity_s2i(state, L, space)
        idxs[invalid] = -1
        return idxs

class Auto(Subspace):
    '''
    Automatically generate a mapping that takes advantage of any possible spin conservation
    law, by performing a breadth-first search of the graph of possible states using the operator
    as an adjacency matrix. The subspace is defined by providing a "start" state; the returned
    subspace will be whatever subspace contains that state.

    This class can provide scalability advantages over the other subspace classes, because it uses
    Cuthill-McKee ordering to reduce the bandwidth of the matrix, reducing the amount of
    communication required between MPI nodes. However, currently the actual computation of the
    ordering only can occur on process 0.

    Parameters
    ----------
    space : int
        An integer whose binary representation corresponds to the spin configuration of the "start"
        state mentioned above.
    '''

    @classmethod
    def _get_dimension(cls, L, space):
        raise NotImplementedError()

    @classmethod
    def _check_space(cls, value):
        raise NotImplementedError()

    @classmethod
    def _check_L(cls, L, space):
        # any L that passes our normal validation checks works
        return L

    @classmethod
    def _idx_to_state(cls, idx, L, space):
        raise NotImplementedError()

    @classmethod
    def _state_to_idx(cls, state, L, space):
        raise NotImplementedError()
