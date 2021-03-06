'''
Classes that define the various subspaces on which operators can be defined.

The methods generally are just an interface to the backend, so that there is only
one implementation of each of the functions.
'''

import numpy as np
from copy import deepcopy
from zlib import crc32

from . import validate, states
from ._backend import bsubspace

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

    def to_enum(self):
        '''
        Convert the class types used in the Python frontend to the enum values
        used in the C backend.
        '''
        raise NotImplementedError()

class Full(Subspace):

    def __init__(self):
        Subspace.__init__(self)

    # Full is a special case
    def __eq__(self, s):
        if isinstance(s, Full):
            return s.L == self.L

        return Subspace.__eq__(self, s)

    # overriding __eq__ causes this to get unset. :(
    __hash__ = Subspace.__hash__

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
        return self._get_cdata(self.L)

    @classmethod
    def _get_cdata(cls, L):
        return bsubspace.CFull(L)

    def to_enum(self):
        '''
        Convert the class types used in the Python frontend to the enum values
        used in the C backend.
        '''
        return bsubspace.SubspaceType.FULL

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
        return self._get_cdata(self.L, self.space)

    @classmethod
    def _get_cdata(cls, L, space):
        return bsubspace.CParity(L, space)

    def to_enum(self):
        '''
        Convert the class types used in the Python frontend to the enum values
        used in the C backend.
        '''
        return bsubspace.SubspaceType.PARITY

class Auto(Subspace):
    '''
    Automatically generate a mapping that takes advantage of any possible spin conservation
    law, by performing a breadth-first search of the graph of possible states using the operator
    as an adjacency matrix. The subspace is defined by providing a "start" state; the returned
    subspace will be whatever subspace contains that state.

    Currently the actual computation of the ordering only can occur on process 0, limiting
    the scalability of this subspace.

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

    sort : bool
        Whether to reorder the mapping after computing it. In some cases this may
        cause a speedup.
    '''

    def __init__(self, H, state, size_guess=None, sort=True):

        Subspace.__init__(self)

        self._L = H.get_length()

        self.state = states.State.str_to_state(state, self.L)

        if size_guess is None:
            size_guess = 2**H.get_length()

        self.state_map = np.ndarray((size_guess,), dtype=bsubspace.dnm_int_t)

        H.reduce_msc()

        dim = bsubspace.compute_rcm(H.msc['masks'], H.msc['signs'], H.msc['coeffs'],
                                    self.state_map, self.state, H.get_length())

        self.state_map = self.state_map[:dim]

        self.rmap_indices = np.argsort(self.state_map).astype(bsubspace.dnm_int_t, copy=False)
        self.rmap_states = self.state_map[self.rmap_indices]
        if sort:
            self.state_map = self.rmap_states
            self.rmap_indices = np.arange(self.state_map.size, dtype=bsubspace.dnm_int_t)

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
        return bsubspace.CAuto(
            self.L,
            np.ascontiguousarray(self.state_map),
            np.ascontiguousarray(self.rmap_indices),
            np.ascontiguousarray(self.rmap_states)
        )

    def to_enum(self):
        '''
        Convert the class types used in the Python frontend to the enum values
        used in the C backend.
        '''
        return bsubspace.SubspaceType.AUTO
