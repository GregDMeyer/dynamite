'''
Classes that define the various subspaces on which operators can be defined.

The methods generally are just an interface to the backend, so that there is only
one implementation of each of the functions.
'''

import numpy as np
from copy import deepcopy
from zlib import crc32
import math
from functools import wraps

from . import validate, states, config
from ._backend import bsubspace
from .msc_tools import dnm_int_t, combine_and_sort
from .bitwise import parity


class Subspace:
    '''
    Base subspace class.
    '''

    # subclasses should set to False if they need
    _product_state_basis = True
    _checksum_start = 0

    # enum value used in the backend to identify subspace
    _enum = None

    # functions each subclass should supply
    _c_get_dimension = None
    _c_idx_to_state = None
    _c_state_to_idx = None

    def __init__(self, L=None):
        if L is None:
            self._L = config.L
        else:
            self._L = validate.L(L)

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
        return hash(self) == hash(s)

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

    @property
    def product_state_basis(self):
        """
        A boolean value indicating whether the given subspace's basis
        states are product states.
        """
        return self._product_state_basis

    def reduce_msc(self, msc, check_conserves=False):
        """
        Return an equivalent (in the subspace) but simpler MSC representation
        for the operator, by taking advantage of the subspace's symmetries.

        Parameters
        ----------

        msc : dynamite.msc_tools.msc_dtype
            The input MSC representation

        check_conserves : bool
            Whether to return whether the operator was conserved

        Returns
        -------

        dynamite.msc_tools.msc_dtype
            The reduced version

        bool
            Whether the operator was conserved during the operation
        """
        raise NotImplementedError

    @classmethod
    def _numeric_to_array(cls, x):
        '''
        Convert numeric values of any type to the type expected by the backend
        functions.
        '''
        x = np.array(x, copy = False, dtype = bsubspace.dnm_int_t).reshape((-1,))
        return np.ascontiguousarray(x)

    def _check_idx_bounds(self, idx):
        dim = self.get_dimension()
        if np.min(idx) < 0 or np.max(idx) >= dim:
            # do this inside the if statement for performance
            invalid_idx = idx[np.logical_or(idx >= dim, idx < 0)]
            if len(invalid_idx) == 1:
                msg = f'Index {invalid_idx[0]}'
            else:
                msg = f'Indices {invalid_idx}'
            msg += f' out of bounds for subspace of dimension {dim}'
            raise ValueError(msg)

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

    def get_dimension(self):
        """
        Get the dimension of the subspace.
        """
        return self._c_get_dimension(self._get_cdata())

    def _single_or_array(fn):
        '''
        Takes a functions that takes and returns arrays, and allows it to take and
        return just a single value as well.
        '''
        @wraps(fn)
        def rtn_fn(self, val, *args, **kwargs):
            single_value = not hasattr(val, "__len__")
            val = self._numeric_to_array(val)

            rtn = fn(self, val, *args, **kwargs)

            if single_value:
                return rtn[0]
            else:
                return rtn

        return rtn_fn

    @_single_or_array
    def idx_to_state(self, idx):
        """
        Maps an index to an integer that in binary corresponds to the spin configuration.
        Vectorized implementation allows passing a numpy array of indices as idx.
        """
        self._check_idx_bounds(idx)
        return self._c_idx_to_state(idx, self._get_cdata())

    @_single_or_array
    def state_to_idx(self, state):
        """
        The inverse mapping of :meth:`idx_to_state`.
        """
        return self._c_state_to_idx(state, self._get_cdata())

    def _to_c(self):
        '''
        Returns the subspace type and data, in C-accessible format.
        '''
        return {'type': self._enum, 'data': self._get_cdata()}

    def _get_cdata(self):
        '''
        Returns an object containing the subspace data accessible by the backend C.
        '''
        raise NotImplementedError()


class Full(Subspace):

    _enum = bsubspace.SubspaceType.FULL
    _c_get_dimension = bsubspace.get_dimension_Full
    _c_idx_to_state = bsubspace.idx_to_state_Full
    _c_state_to_idx = bsubspace.state_to_idx_Full

    def __init__(self, L=None):
        Subspace.__init__(self, L)

    def __eq__(self, s):
        if isinstance(s, Full):
            return s.L == self.L

        return Subspace.__eq__(self, s)

    def __hash__(self):
        return hash((self._enum, self.L))

    def __repr__(self):
        if self.L is None:
            arg_str = ''
        else:
            arg_str = f'L={self.L}'

        return f'Full({arg_str})'

    def _get_cdata(self):
        '''
        Returns an object containing the subspace data accessible by the C backend.
        '''
        if self.L is None:
            raise ValueError('L has not been set for this subspace')
        return bsubspace.CFull(self.L)


class Parity(Subspace):
    '''
    The subspaces of states in which the number of up spins is even or odd.

    Parameters
    ----------
    space : int
        Either 0 or 'even' for the even subspace, or 1 or 'odd' for the other.
    '''

    _enum = bsubspace.SubspaceType.PARITY
    _c_get_dimension = bsubspace.get_dimension_Parity
    _c_idx_to_state = bsubspace.idx_to_state_Parity
    _c_state_to_idx = bsubspace.state_to_idx_Parity

    def __init__(self, space, L=None):
        Subspace.__init__(self, L)
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

    def __hash__(self):
        return hash((self._enum, self.L, self.space))

    def __repr__(self):
        arg_str = {0: "'even'", 1: "'odd'"}[self.space]
        if self.L is not None:
            arg_str += f', L={self.L}'

        return f'Parity({arg_str})'

    def _get_cdata(self):
        '''
        Returns an object containing the subspace data accessible by the C backend.
        '''
        if self.L is None:
            raise ValueError('L has not been set for this subspace')
        return bsubspace.CParity(self.L, self.space)


class SpinConserve(Subspace):
    '''
    The subspaces of states which conserve total magnetization (total
    number of up/down spins).

    Parameters
    ----------
    L : int
        Length of spin chain

    k : int
        Number of down spins (1's in integer representation of state)

    spinflip : str, optional
        Sign of spinflip basis ('+' or '-'). Omit to not use Z2 symmetry.
    '''

    _product_state_basis = False
    _enum = bsubspace.SubspaceType.SPIN_CONSERVE
    _c_get_dimension = bsubspace.get_dimension_SpinConserve
    _c_idx_to_state = bsubspace.idx_to_state_SpinConserve
    _c_state_to_idx = bsubspace.state_to_idx_SpinConserve

    def __init__(self, L, k, spinflip=None):
        Subspace.__init__(self, L=L)

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

    def __hash__(self):
        return hash((self._enum, self.L, self.k, self.spinflip))

    def __repr__(self):
        arg_str = f'L={self.L}, k={self.k}'
        if self.spinflip != 0:
            arg_str += f', spinflip={self.spinflip:+d}'
        return f'SpinConserve({arg_str})'

    def reduce_msc(self, msc, check_conserves=False):
        if self.spinflip == 0:
            raise NotImplementedError("reduce_msc should not be called when "
                                      "not using the additional spinflip "
                                      "subspace")

        msc = msc.copy()

        # delete elements which do not commute with symmetry operator
        keep = parity(msc['signs']) == 0
        conserved = np.all(keep)
        msc = msc[keep]

        terms_to_mod = np.nonzero(msc['masks'] >> (self.L-1))
        msc['masks'][terms_to_mod] ^= (1 << self.L) - 1

        if self.spinflip == -1:
            msc['coeffs'][terms_to_mod] *= -1

        msc = combine_and_sort(msc)

        if check_conserves:
            return msc, conserved
        else:
            return msc

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
        state.assert_initialized()

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

    @property
    def k(self):
        """
        The number of up ("0") spins.
        """
        return self._k

    def _get_cdata(self):
        '''
        Returns an object containing the subspace data accessible by the C backend.
        '''
        return bsubspace.CSpinConserve(
            self.L, self.k,
            np.ascontiguousarray(self._nchoosek),
            self.spinflip
        )


class Explicit(Subspace):
    '''
    A subspace generated by explicitly passing a list of product states.

    Parameters
    ----------
    state_list : array-like
        An array of integers representing the states (in binary).
    '''

    _enum = bsubspace.SubspaceType.EXPLICIT
    _c_get_dimension = bsubspace.get_dimension_Explicit
    _c_idx_to_state = bsubspace.idx_to_state_Explicit
    _c_state_to_idx = bsubspace.state_to_idx_Explicit

    def __init__(self, state_list, L=None):
        Subspace.__init__(self, L=L)
        self.state_map = np.asarray(state_list, dtype=bsubspace.dnm_int_t)

        map_sorted = np.all(self.state_map[:-1] <= self.state_map[1:])

        if map_sorted:
            self.rmap_indices = np.array([-1], dtype=bsubspace.dnm_int_t)
            self.rmap_states = self.state_map
        else:
            self.rmap_indices = np.argsort(self.state_map).astype(bsubspace.dnm_int_t, copy=False)
            self.rmap_states = self.state_map[self.rmap_indices]

        if L is not None:
            self.check_L(L)

    def check_L(self, value):
        # last value of rmap_states is the lexicographically largest one
        if self.rmap_states[-1] >> value != 0:
            raise ValueError('State in subspace has more spins than provided')
        return value

    def __hash__(self):
        return hash((self._enum, self.get_checksum()))

    def __repr__(self):
        arg_str = f'{self.state_map}'
        if self.L is not None:
            arg_str += f', L={self.L}'
        return f'Explicit({arg_str})'

    def _get_cdata(self):
        '''
        Returns an object containing the subspace data accessible by the C backend.
        '''
        return bsubspace.CExplicit(
            self.L,
            np.ascontiguousarray(self.state_map),
            np.ascontiguousarray(self.rmap_indices),
            np.ascontiguousarray(self.rmap_states)
        )


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

        # construct repr args it now so we don't have to save a ton of stuff
        self._repr_args = f'H={repr(H)}, state={repr(state)}'
        if size_guess is not None:
            self._repr_args += f', size_guess={size_guess}'
        if not sort:
            self._repr_args += ', sort=False'

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

        Explicit.__init__(self, state_map, L=H.L)

    def __repr__(self):
        return f'Auto({self._repr_args})'
