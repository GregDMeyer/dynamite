'''
Classes that define the various subspaces on which operators can be defined.

The methods generally are just an interface to the backend, so that there is only
one implementation of each of the functions.
'''

import numpy as np
from copy import deepcopy
from zlib import crc32
import math

from . import validate, states, config
from ._backend import bsubspace
from .msc_tools import dnm_int_t

class Subspace:
    '''
    Base subspace class.
    '''

    # subclasses should set to False if they need
    _product_state_basis = True
    _checksum_start = 0

    def __init__(self):
        self._L = config.L
        self._chksum = None

    def __eq__(self, s):
        '''
        Returns true if the two subspaces correspond to the same mapping, even if they
        are different classes.
        '''
        if s is self:
            return True

        if not isinstance(s, Subspace):
            raise ValueError('Cannot compare Subspace to non-Subspace type')

        if self._L is None:
            raise ValueError('Cannot evaluate equality of subspaces before setting L')

        if self.get_dimension() != s.get_dimension():
            return False

        return self.get_checksum() == s.get_checksum()

    def identical(self, s):
        '''
        Returns whether two subspaces are exactly the same---both the same
        type and with the same values.
        '''
        raise NotImplementedError()

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
        if self.L is not None and value != self.L:
            raise AttributeError('Cannot change L for a subspace after it is set')

        # check that this value of L is compatible with the subspace
        value = validate.L(value)
        value = self.check_L(value)
        self._L = value

    def get_dimension(self):
        """
        Get the dimension of the subspace.
        """
        raise NotImplementedError()

    @property
    def product_state_basis(self):
        """
        A boolean value indicating whether the given subspace's basis
        states are product states.
        """
        return self._product_state_basis

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
            chksum = self._checksum_start
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

    def identical(self, s):
        return type(self) == type(s) and self.L == s.L

    def get_dimension(self):
        """
        Get the dimension of the subspace.
        """
        if self.L is None:
            raise ValueError('L has not been set for this subspace')
        return self._get_dimension(self.L)

    @classmethod
    def _get_dimension(cls, L):
        return bsubspace.get_dimension_Full(cls._get_cdata(L))

    def idx_to_state(self, idx):
        """
        Maps an index to an integer that in binary corresponds to the spin configuration.
        Vectorized implementation allows passing a numpy array of indices as idx.
        """
        if self.L is None:
            raise ValueError('L has not been set for this subspace')
        return self._idx_to_state(idx, self.L)

    def state_to_idx(self, state):
        """
        The inverse mapping of :meth:`idx_to_state`.
        """
        if self.L is None:
            raise ValueError('L has not been set for this subspace')
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
        if self.L is None:
            raise ValueError('L has not been set for this subspace')
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

    def identical(self, s):
        if type(self) != type(s):
            return False

        if self.L != s.L:
            return False

        if self.space != s.space:
            return False

        return True

    def get_dimension(self):
        """
        Get the dimension of the subspace.
        """
        if self.L is None:
            raise ValueError('L has not been set for this subspace')
        return self._get_dimension(self.L, self.space)

    @classmethod
    def _get_dimension(cls, L, space):
        return bsubspace.get_dimension_Parity(cls._get_cdata(L, space))

    def idx_to_state(self, idx):
        """
        Maps an index to an integer that in binary corresponds to the spin configuration.
        Vectorized implementation allows passing a numpy array of indices as idx.
        """
        if self.L is None:
            raise ValueError('L has not been set for this subspace')
        idx = self._numeric_to_array(idx)
        return self._idx_to_state(idx, self.L, self.space)

    def state_to_idx(self, state):
        """
        The inverse mapping of :meth:`idx_to_state`.
        """
        if self.L is None:
            raise ValueError('L has not been set for this subspace')
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
        if self.L is None:
            raise ValueError('L has not been set for this subspace')
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


class SpinConserve(Subspace):
    '''
    The subspaces of states which conserve total magnetization (total
    number of up/down spins).

    Parameters
    ----------
    L : int
        Length of spin chain (constant for this class)

    k : int
        Number of down spins (1's in integer representation of state)

    spinflip : str, optional
        Sign of spinflip basis ('+' or '-'). Omit to not use Z2 symmetry.
    '''

    _product_state_basis = False

    def __init__(self, L, k, spinflip=None):
        Subspace.__init__(self)
        self._L = validate.L(L)

        if spinflip is None or spinflip == 0:
            self._spinflip = 0
        elif spinflip in ['+', +1]:
            self._spinflip = +1
        elif spinflip in ['-', -1]:
            self._spinflip = -1
        else:
            raise ValueError('invalid value for spinflip')

        self._product_state_basis = self.spinflip == 0

        # unique checksum for each value of spinflip
        self._checksum_start = self._spinflip

        self._k = self._check_k(k)

        self._nchoosek = self._compute_nchoosek(L, k)

    def identical(self, s):
        if type(self) != type(s):
            return False

        if self.L != s.L or self.k != s.k:
            return False

        if self.spinflip != s.spinflip:
            return False

        return True

    @property
    def spinflip(self):
        """
        Whether the subspace uses the additional spinflip symmetry.
        Returns integer +1, -1, or 0 (no spinflip symmetry).
        """
        return self._spinflip

    @classmethod
    def convert_spinflip(cls, state, sign=None):
        """
        Convert a state on a subspace where spinflip is set
        to a state on a product state SpinConserve subspace,
        and vice versa

        Parameters
        ----------

        state : State
            The input state

        sign : str, optional
            The sign of the spinflip subspace. Required when converting from
            non-spinflip subspace.

        Returns
        -------

        State
            The converted state
        """
        if not state.initialized:
            # this import has to be done here to avoid circular import
            from .states import UninitializedError
            raise UninitializedError("State vector data has not been set yet")

        if state.subspace.spinflip == 0 and sign is None:
            raise ValueError('must provide sign when converting to spinflip')

        if state.subspace.spinflip != 0 and sign is not None:
            raise ValueError("do not provide sign when converting from "
                             "spinflip subspace")

        if sign in ['+', +1]:
            sign = +1
        elif sign in ['-', -1]:
            sign = -1
        elif sign is not None:
            raise ValueError('invalid value of sign')

        new_space = SpinConserve(state.subspace.L,
                                 state.subspace.k,
                                 spinflip=sign)

        rtn_state = states.State(subspace=new_space)
        istart, iend = state.vec.getOwnershipRange()

        n_in = len(state)
        n_rtn = len(rtn_state)
        if state.subspace.spinflip:  # to product state basis
            start = n_rtn-istart-1
            end = n_rtn-iend-1
            rtn_state.vec[start:end:-1] = state.vec[istart:iend]

            if state.subspace.spinflip == -1:
                # flip sign of second half of vector for - subspace
                rtn_state.vec.assemble()
                rtn_state.vec.scale(-1)

            rtn_state.vec[istart:iend] = state.vec[istart:iend]

        else:  # from product state basis

            # second half of vector
            start = max(n_in//2, istart)
            end = iend
            if start < end:
                # an unfortunate side effect of python slice notation
                if n_in == end:
                    rtn_state.vec[n_in-start-1::-1] = state.vec[start:end]
                else:
                    rtn_state.vec[n_in-start-1:n_in-end-1:-1] = state.vec[start:end]

            rtn_state.vec.assemble()
            if sign == -1:
                rtn_state.vec.scale(-1)

            if istart < n_in//2:
                start = istart
                end = min(n_in//2, iend)
                rtn_state.vec.setValues(np.arange(start, end, dtype=dnm_int_t),
                                        state.vec[start:end],
                                        addv=True)

        rtn_state.vec.assemble()
        rtn_state.vec.scale(1/np.sqrt(2))

        rtn_state.set_initialized()

        return rtn_state

    @classmethod
    def _compute_nchoosek(cls, L, k):
        # we index over k first to hopefully make the memory access pattern
        # slightly better. sorry :-(
        rtn = np.ndarray((k+1, L+1), dtype=bsubspace.dnm_int_t)

        # there is a more efficient algorithm where we use a combinations
        # identity to compute the values way faster. but it's super fast anyway
        # and this is more readable
        for (kk, LL), _ in np.ndenumerate(rtn):
            rtn[kk, LL] = math.comb(LL, kk)

        return rtn

    def _check_k(self, k):
        if not (0 <= k <= self.L):
            raise ValueError('k must be between 0 and L')

        if self._spinflip and not 2*k == self.L:
            raise ValueError('L must equal 2k for spinflip symmetry')

        return k

    @Subspace.L.setter
    def L(self, value):
        if value != self.L:
            raise AttributeError('cannot change L for SpinConserve class')

    @property
    def k(self):
        """
        The number of up ("0") spins.
        """
        return self._k

    def get_dimension(self):
        """
        Get the dimension of the subspace.
        """
        return self._get_dimension(self.L, self.k, self._nchoosek, self._spinflip)

    @classmethod
    def _get_dimension(cls, L, k, nchoosek, spinflip=0):
        return bsubspace.get_dimension_SpinConserve(cls._get_cdata(L, k, nchoosek, spinflip))

    def idx_to_state(self, idx):
        """
        Maps an index to an integer that in binary corresponds to the spin configuration.
        Vectorized implementation allows passing a numpy array of indices as idx.
        """
        idx = self._numeric_to_array(idx)
        return self._idx_to_state(idx, self.L, self.k, self._nchoosek, self._spinflip)

    def state_to_idx(self, state):
        """
        The inverse mapping of :meth:`idx_to_state`.
        """
        state = self._numeric_to_array(state)
        return self._state_to_idx(state, self.L, self.k, self._nchoosek, self._spinflip)

    @classmethod
    def _idx_to_state(cls, idx, L, k, nchoosek, spinflip=0):
        return bsubspace.idx_to_state_SpinConserve(idx, cls._get_cdata(L, k, nchoosek, spinflip))

    @classmethod
    def _state_to_idx(cls, state, L, k, nchoosek, spinflip=0):
        return bsubspace.state_to_idx_SpinConserve(state, cls._get_cdata(L, k, nchoosek, spinflip))

    def get_cdata(self):
        '''
        Returns an object containing the subspace data accessible by the C backend.
        '''
        return self._get_cdata(self.L, self.k, self._nchoosek, self._spinflip)

    @classmethod
    def _get_cdata(cls, L, k, nchoosek, spinflip=0):
        return bsubspace.CSpinConserve(
            L, k,
            np.ascontiguousarray(nchoosek),
            spinflip
        )

    def to_enum(self):
        '''
        Convert the class types used in the Python frontend to the enum values
        used in the C backend.
        '''
        return bsubspace.SubspaceType.SPIN_CONSERVE


class Explicit(Subspace):
    '''
    A subspace generated by explicitly passing a list of product states.

    Parameters
    ----------
    state_list : array-like
        An array of integers representing the states (in binary).
    '''

    def __init__(self, state_list):
        Subspace.__init__(self)
        self.state_map = np.asarray(state_list, dtype=bsubspace.dnm_int_t)

        map_sorted = np.all(self.state_map[:-1] <= self.state_map[1:])

        if map_sorted:
            self.rmap_indices = np.arange(self.state_map.size, dtype=bsubspace.dnm_int_t)
            self.rmap_states = self.state_map
        else:
            self.rmap_indices = np.argsort(self.state_map).astype(bsubspace.dnm_int_t, copy=False)
            self.rmap_states = self.state_map[self.rmap_indices]

    def check_L(self, value):
        # last value of rmap_states is the lexicographically largest one
        if self.rmap_states[-1] >> value != 0:
            raise ValueError('State in subspace has more spins than provided')
        return value

    def identical(self, s):
        if type(self) != type(s):
            return False

        return self == s

    def get_dimension(self):
        """
        Get the dimension of the subspace.
        """
        return bsubspace.get_dimension_Explicit(self.get_cdata())

    def idx_to_state(self, idx):
        """
        Maps an index to an integer that in binary corresponds to the spin configuration.
        Vectorized implementation allows passing a numpy array of indices as idx.
        """
        idx = self._numeric_to_array(idx)
        return bsubspace.idx_to_state_Explicit(idx, self.get_cdata())

    def state_to_idx(self, state):
        """
        The inverse mapping of :meth:`idx_to_state`.
        """
        state = self._numeric_to_array(state)
        return bsubspace.state_to_idx_Explicit(state, self.get_cdata())

    def get_cdata(self):
        '''
        Returns an object containing the subspace data accessible by the C backend.
        '''
        return bsubspace.CExplicit(
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
        return bsubspace.SubspaceType.EXPLICIT


class Auto(Explicit):
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

        H.establish_L()

        self.state = states.State.str_to_state(state, H.L)

        if size_guess is None:
            size_guess = 2**H.L

        state_map = np.ndarray((size_guess,), dtype=bsubspace.dnm_int_t)

        H.reduce_msc()

        dim = bsubspace.compute_rcm(H.msc['masks'], H.msc['signs'], H.msc['coeffs'],
                                    state_map, self.state, H.L)

        state_map = state_map[:dim]

        if sort:
            state_map.sort()

        Explicit.__init__(self, state_map)

        self._L = H.L
