'''
Classes that define the various subspaces on which operators can be defined.

The methods generally are just an interface to the backend, so that there is only
one implementation of each of the functions.
'''

import numpy as np
from copy import deepcopy
from zlib import crc32

from . import validate, info, states
from ._backend import bsubspace

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
    '''
    Base subspace class.
    '''

    def __init__(self):
        self._L = None
        self._chksum = None

    def __eq__(self, s):
        '''
        Returns true if the two subspaces correspond to the same mapping, even if they
        are different classes.
        '''
        if not isinstance(s, Subspace):
            raise ValueError('Cannot compare Subspace to non-Subspace type')

        if self.get_dimension() != s.get_dimension():
            return False

        return self.get_checksum() == s.get_checksum()

    @property
    def L(self):
        '''
        The spin chain length corresponding to this space.
        '''
        return self._L

    def check_L(self, value):
        # by default, any L that passes our normal validation checks works
        return value

    @L.setter
    def L(self, value):
        # check that this value of L is compatible with the subspace
        value = validate.L(value)
        value = self.check_L(value)

        if value != self._L:
            self._chksum = None

        self._L = value

    def get_dimension(self):
        """
        Get the dimension of the subspace.
        """
        raise NotImplementedError()

    @classmethod
    def _numeric_to_array(cls, x):
        '''
        Convert numeric values of any type to the type expected by the backend
        functions.
        '''
        x = np.array(x, copy = False, dtype = bsubspace.dnm_int_t).reshape((-1,))
        return np.ascontiguousarray(x)

    def idx_to_state(self, idx):
        """
        Maps an index to an integer that in binary corresponds to the spin configuration.
        Vectorized implementation allows passing a numpy array of indices as idx.
        """
        raise NotImplementedError()

    def state_to_idx(self, state):
        """
        The inverse mapping of :meth:`idx_to_state`.
        """
        raise NotImplementedError()

    def copy(self):
        return deepcopy(self)

    def get_checksum(self):
        '''
        Get a checksum of the state mapping for this subspace. This allows subspaces to
        be compared quickly.
        '''
        if self._chksum is None:
            BLOCK = 2**14
            chksum = 0
            for start in range(0, self.get_dimension(), BLOCK):
                stop = min(start+BLOCK, self.get_dimension())
                smap = self.idx_to_state(np.arange(start, stop))
                chksum = crc32(smap, chksum)
            self._chksum = chksum

        return self._chksum

    def __hash__(self):
        return self.get_checksum()

    def get_cdata(self):
        '''
        Returns an object containing the subspace data accessible by the backend C.
        '''
        raise NotImplementedError()

class Full(Subspace):

    def __init__(self):
        Subspace.__init__(self)

    def get_dimension(self):
        """
        Get the dimension of the subspace.
        """
        return self._get_dimension(self.L)

    @classmethod
    def _get_dimension(cls, L):
        return bsubspace.get_dimension_Full(cls._get_cdata(L))

    def idx_to_state(self, idx):
        """
        Maps an index to an integer that in binary corresponds to the spin configuration.
        Vectorized implementation allows passing a numpy array of indices as idx.
        """
        return self._idx_to_state(idx, self.L)

    def state_to_idx(self, state):
        """
        The inverse mapping of :meth:`idx_to_state`.
        """
        return self._state_to_idx(state, self.L)

    @classmethod
    def _idx_to_state(cls, idx, L):
        idx = cls._numeric_to_array(idx)
        return bsubspace.idx_to_state_Full(idx, cls._get_cdata(L))

    @classmethod
    def _state_to_idx(cls, state, L):
        state = cls._numeric_to_array(state)
        return bsubspace.state_to_idx_Full(state, cls._get_cdata(L))

    def get_cdata(self):
        '''
        Returns an object containing the subspace data accessible by the C backend.
        '''
        info.write(2, 'Getting C subspace data for Full subspace.')
        return self._get_cdata(self.L)

    @classmethod
    def _get_cdata(cls, L):
        return bsubspace.CFull(L)

class Parity(Subspace):
    '''
    The subspaces of states in which the number of up spins is even or odd.

    Parameters
    ----------
    space : int
        Either 0 or 'even' for the even subspace, or 1 or 'odd' for the other.
    '''

    def __init__(self, space):
        Subspace.__init__(self)
        self._space = self._check_space(space)

    @property
    def space(self):
        return self._space

    @classmethod
    def _check_space(cls, value):
        if value in [0,'even']:
            return 0
        elif value in [1,'odd']:
            return 1
        else:
            raise ValueError('Invalid parity space "'+str(value)+'" '
                             '(valid choices are 0, 1, "even", or "odd")')

    def get_dimension(self):
        """
        Get the dimension of the subspace.
        """
        return self._get_dimension(self.L, self.space)

    @classmethod
    def _get_dimension(cls, L, space):
        return bsubspace.get_dimension_Parity(cls._get_cdata(L, space))

    def idx_to_state(self, idx):
        """
        Maps an index to an integer that in binary corresponds to the spin configuration.
        Vectorized implementation allows passing a numpy array of indices as idx.
        """
        idx = self._numeric_to_array(idx)
        return self._idx_to_state(idx, self.L, self.space)

    def state_to_idx(self, state):
        """
        The inverse mapping of :meth:`idx_to_state`.
        """
        state = self._numeric_to_array(state)
        return self._state_to_idx(state, self.L, self.space)

    @classmethod
    def _idx_to_state(cls, idx, L, space):
        return bsubspace.idx_to_state_Parity(idx, cls._get_cdata(L, space))

    @classmethod
    def _state_to_idx(cls, state, L, space):
        return bsubspace.state_to_idx_Parity(state, cls._get_cdata(L, space))

    def get_cdata(self):
        '''
        Returns an object containing the subspace data accessible by the C backend.
        '''
        info.write(2, 'Getting C subspace data for Parity subspace.')
        return self._get_cdata(self.L, self.space)

    @classmethod
    def _get_cdata(cls, L, space):
        return bsubspace.CParity(L, space)

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
    H : dynamite.operators.Operator
        The operator for which this custom subspace will be defined.

    state : int or string
        An integer whose binary representation corresponds to the spin configuration of the "start"
        state mentioned above, or string representing the same. See
        :meth:`dynamite.states.State.str_to_state` for more information.

    size_guess : int
        A guess for the dimension of the subspace. By default, memory is allocated for the full
        space, and then trimmed off if not used.
    '''

    def __init__(self, H, state, size_guess=None):

        Subspace.__init__(self)

        self._L = H.get_length()

        self.state = states.State.str_to_state(state, self.L)

        if size_guess is None:
            size_guess = 2**H.get_length()

        self.state_map = np.ndarray((size_guess,), dtype=bsubspace.dnm_int_t)

        # TODO: use some sort of custom hash table for this data structure?
        # python's dict won't work because it's not vectorized, and too slow anyway
        self.state_rmap = np.ndarray((2**H.get_length(),), dtype=bsubspace.dnm_int_t)
        self.state_rmap[:] = -1

        H.reduce_msc()

        dim = bsubspace.compute_rcm(H.msc['masks'], H.msc['signs'], H.msc['coeffs'],
                                    self.state_map, self.state_rmap, self.state, H.get_length())

        self.state_map = self.state_map[:dim]

    def check_L(self, value):
        if value != self.L:
            raise TypeError('Cannot change L for Auto subspace type.')
        return value

    def get_dimension(self):
        """
        Get the dimension of the subspace.
        """
        return bsubspace.get_dimension_Auto(self.get_cdata())

    def idx_to_state(self, idx):
        """
        Maps an index to an integer that in binary corresponds to the spin configuration.
        Vectorized implementation allows passing a numpy array of indices as idx.
        """
        idx = self._numeric_to_array(idx)
        return bsubspace.idx_to_state_Auto(idx, self.get_cdata())

    def state_to_idx(self, state):
        """
        The inverse mapping of :meth:`idx_to_state`.
        """
        state = self._numeric_to_array(state)
        return bsubspace.state_to_idx_Auto(state, self.get_cdata())

    def get_cdata(self):
        '''
        Returns an object containing the subspace data accessible by the C backend.
        '''
        info.write(2, 'Getting C subspace data for Auto subspace.')
        return bsubspace.CAuto(np.ascontiguousarray(self.state_map),
                               np.ascontiguousarray(self.state_rmap))
