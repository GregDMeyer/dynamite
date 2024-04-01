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

from . import validate, config, states
from ._backend import bsubspace
from .msc_tools import dnm_int_t, combine_and_sort
from .bitwise import parity


class Subspace:
    '''
    Base subspace class.
    '''

    _chksum = None

    def __eq__(self, s):
        '''
        Returns true if the two subspaces correspond to the same mapping, even if they
        are different classes.
        '''
        if s is self:
            return True

        if not isinstance(s, Subspace):
            raise ValueError('Cannot compare Subspace to non-Subspace type')

        if self.L is None:
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

    def get_dimension(self):
        """
        Get the dimension of the subspace.
        """
        return self._get_dimension()

    def _get_dimension(self):
        raise NotImplementedError

    def _single_or_array(fn):
        '''
        Takes a functions that takes and returns arrays, and allows it to take and
        return just a single value as well.
        '''
        @wraps(fn)
        def rtn_fn(self, val, *args, **kwargs):
            single_value = not hasattr(val, "__len__")
            val = np.ascontiguousarray(
                np.array(val, copy=False, dtype=bsubspace.dnm_int_t).reshape((-1,))
            )

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

        # check that all indices are in bounds
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

        return self._idx_to_state(idx)

    def _idx_to_state(idx):
        raise NotImplementedError

    @_single_or_array
    def state_to_idx(self, state):
        """
        The inverse mapping of :meth:`idx_to_state`.
        """
        return self._state_to_idx(state)

    def _state_to_idx(self, state):
        raise NotImplementedError

    def _to_c(self):
        '''
        Returns a dict with the fields 'type' and 'data' containing information about the underlying
        product state subspace. Any subspace on top of that (currently only XParity) needs to be
        handled separately.
        '''
        raise NotImplementedError


class _ProductStateSubspace(Subspace):
    """
    A subspace whose basis states are product states in the Z basis. Subspaces of this
    class underlie the non-product-state subspaces (currently only XParity).
    """
    _product_state_basis = True

    # enum value used in the backend to identify subspace
    _enum = None

    # functions each subclass should supply
    _c_get_dimension = None
    _c_idx_to_state = None
    _c_state_to_idx = None

    def __init__(self, L=None):
        self._L = None
        if L is None:
            L = config.L

        if L is not None:
            self.L = L

    def _get_dimension(self):
        return self._c_get_dimension(self._get_cdata())

    def _idx_to_state(self, idx):
        return self._c_idx_to_state(idx, self._get_cdata())

    def _state_to_idx(self, state):
        return self._c_state_to_idx(state, self._get_cdata())

    def _get_cdata(self):
        '''
        Returns an object containing the subspace data accessible by the backend C.
        '''
        raise NotImplementedError()

    def _to_c(self):
        return {'type': self._enum, 'data': self._get_cdata()}


class Full(_ProductStateSubspace):

    _enum = bsubspace.SubspaceType.FULL
    _c_get_dimension = staticmethod(bsubspace.get_dimension_Full)
    _c_idx_to_state = staticmethod(bsubspace.idx_to_state_Full)
    _c_state_to_idx = staticmethod(bsubspace.state_to_idx_Full)

    def __init__(self, L=None):
        super().__init__(L)

    def __eq__(self, s):
        if isinstance(s, Full):
            return s.L == self.L

        return super().__eq__(s)

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


class Parity(_ProductStateSubspace):
    '''
    The subspaces of states in which the number of up spins is even or odd.

    Parameters
    ----------
    space : int
        Either 0 or 'even' for the even subspace, or 1 or 'odd' for the other.
    '''

    _enum = bsubspace.SubspaceType.PARITY
    _c_get_dimension = staticmethod(bsubspace.get_dimension_Parity)
    _c_idx_to_state = staticmethod(bsubspace.idx_to_state_Parity)
    _c_state_to_idx = staticmethod(bsubspace.state_to_idx_Parity)

    def __init__(self, space, L=None):
        super().__init__(L)
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


class SpinConserve(_ProductStateSubspace):
    '''
    The subspaces of states which conserve total magnetization (total
    number of up/down spins).

    Parameters
    ----------
    L : int
        Length of spin chain

    k : int
        Number of down spins (1's in integer representation of state)

    spinflip : None
        (deprecated, use ``XParity`` subspace)
    '''

    _enum = bsubspace.SubspaceType.SPIN_CONSERVE
    _c_get_dimension = staticmethod(bsubspace.get_dimension_SpinConserve)
    _c_idx_to_state = staticmethod(bsubspace.idx_to_state_SpinConserve)
    _c_state_to_idx = staticmethod(bsubspace.state_to_idx_SpinConserve)

    def __init__(self, L, k, spinflip=None):
        super().__init__(L=L)
        self._k = self._check_k(k)
        self._nchoosek = self._compute_nchoosek(L, k)
        if spinflip is not None:
            raise DeprecationWarning('spinflip argument has been deprecated; use the XParity '
                                     'class instead.')

    def __hash__(self):
        return hash((self._enum, self.L, self.k))

    def __repr__(self):
        return f'SpinConserve(L={self.L}, k={self.k})'

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

        return k

    @property
    def k(self):
        """
        The number of down ("1" in binary representation) spins.
        """
        return self._k

    def _get_cdata(self):
        '''
        Returns an object containing the subspace data accessible by the C backend.
        '''
        if self.L is None:
            raise ValueError('L has not been set for this subspace')

        return bsubspace.CSpinConserve(
            self.L, self.k,
            np.ascontiguousarray(self._nchoosek)
        )


class Explicit(_ProductStateSubspace):
    '''
    A subspace generated by explicitly passing a list of product states.

    Parameters
    ----------
    state_list : array-like
        An array of integers representing the states (in binary).
    '''

    _enum = bsubspace.SubspaceType.EXPLICIT
    _c_get_dimension = staticmethod(bsubspace.get_dimension_Explicit)
    _c_idx_to_state = staticmethod(bsubspace.idx_to_state_Explicit)
    _c_state_to_idx = staticmethod(bsubspace.state_to_idx_Explicit)

    def __init__(self, state_list, L=None):
        self.state_map = np.asarray(state_list, dtype=bsubspace.dnm_int_t)

        map_sorted = np.all(self.state_map[:-1] <= self.state_map[1:])

        if map_sorted:
            self.rmap_indices = np.array([-1], dtype=bsubspace.dnm_int_t)
            self.rmap_states = self.state_map
        else:
            self.rmap_indices = np.argsort(self.state_map).astype(bsubspace.dnm_int_t, copy=False)
            self.rmap_states = self.state_map[self.rmap_indices]

        # ensure all states are unique
        if np.any(self.rmap_states[1:] == self.rmap_states[:-1]):
            raise ValueError('values in state_list must be unique')

        # need to keep a handle on the contiguous versions of these,
        # so they don't get garbage collected
        self.state_map = np.ascontiguousarray(self.state_map)
        self.rmap_indices = np.ascontiguousarray(self.rmap_indices)
        self.rmap_states = np.ascontiguousarray(self.rmap_states)

        super().__init__(L=L)

    def check_L(self, value):
        # last value of rmap_states is the lexicographically largest one
        if self.rmap_states[-1] >> value != 0:
            raise ValueError('State in subspace has more spins than provided')
        return value

    def __hash__(self):
        return hash((self._enum, self.get_checksum()))

    def __repr__(self):
        # following numpy's lead about when to put ellipsis
        if len(self.state_map) < 1000:
            to_show = self.state_map
        else:
            to_show = list(self.state_map[:3]) + ['...'] + list(self.state_map[-3:])

        if self.L is None:
            # number of bits in max value of state map
            L = int(self.rmap_states[-1]).bit_length()
        else:
            L = self.L

        # python 0b... integers, but with zeros filled to length L
        arg_str = '[' + ', '.join(
            x if isinstance(x, str) else '0b' + bin(x)[2:].zfill(L)
            for x in to_show
        ) + ']'

        if self.L is not None:
            arg_str += f', L={self.L}'
        return f'Explicit({arg_str})'

    def _get_cdata(self):
        '''
        Returns an object containing the subspace data accessible by the C backend.
        '''
        if self.L is None:
            raise ValueError('L has not been set for this subspace')

        return bsubspace.CExplicit(
            self.L,
            self.state_map,
            self.rmap_indices,
            self.rmap_states
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
        else:
            # reverse Cuthill-McKee ordering needs... reverse!
            state_map = state_map[::-1]

        Explicit.__init__(self, state_map, L=H.L)

    def __repr__(self):
        return f'Auto({self._repr_args})'


class XParity(Subspace):
    r'''
    This class implements the Parity subspace, but in the X basis instead
    of the Z basis. Unlike the other subspaces, it can be applied on top of
    another subspace by passing that subspace as the ``parent`` argument.

    In the Z basis, the basis states of this subspace are not product states,
    but rather states of the form :math:`\left|c\right> + \left|\bar c \right>`,
    where :math:`\left|c\right>` is a product state and :math:`\left|\bar c\right>`
    is its complement (all spins flipped). In dynamite's interface, these basis states
    are represented by the bitstring :math:`c` or :math:`\bar c` that is lexicographically
    first (that is, the one having spin L-1 in the :math:`\left|0\right>` state).
    '''

    _product_state_basis = False

    def __init__(self, parent=None, sector='+', L=None):
        if parent is None:
            parent = Full()

        self._parent = parent
        if L is not None:
            self.parent.L = L

        self._validate_parent(self.parent)

        if sector in ['+', +1]:
            self._sector = +1
        elif sector in ['-', -1]:
            self._sector = -1
        else:
            raise ValueError('invalid value for sector')

    @classmethod
    def _validate_parent(cls, parent):
        if not parent.product_state_basis:
            raise ValueError('parent must be a product state subspace')

        # Full is always fine
        if isinstance(parent, Full):
            return

        if parent.L is None:
            raise ValueError('L must be set for the parent subspace')

        # Parity is fine if L is even
        if isinstance(parent, Parity):
            if parent.L % 2 == 0:
                return

            raise ValueError('Parity is only compatible with XParity when L is even')

        # SpinConserve is only OK at half filling
        if isinstance(parent, SpinConserve):
            if parent.L == 2*parent.k:
                return

            raise ValueError('SpinConserve is only compatible with XParity when k=L/2')

        # otherwise (currently only could be Explicit) we have to check... explicitly!

        dim = parent.get_dimension()
        if dim % 2 != 0:
            raise ValueError('parent subspace must have even dimension')

        block_size = 1024  # for efficiency
        for start in range(0, dim//2, block_size):
            end = min(start+block_size, dim//2)

            # states in first half, they are representatives
            state_block = parent.idx_to_state(np.arange(start, end))

            # make sure they all start with 0
            if np.count_nonzero(state_block >> (parent.L-1)):
                raise ValueError('first dim/2 basis states must have spin L-1 up '
                                 '(0 in integer notation)')

            # make sure their complements are also in subspace
            if np.any(parent.state_to_idx(state_block) == -1):
                raise ValueError('the complement of every state in subspace (all spins flipped) '
                                 'must also be in subspace')

            # we don't need to check the ones starting with 1, because all basis states are unique,
            # so if half of states start with 0 and each one has a complement there is no room for
            # extra states starting with 1 that don't have a matching 0 state

    @property
    def parent(self):
        """
        The parent subspace upon which XParity has been applied.
        """
        return self._parent

    @property
    def sector(self):
        """
        The sector of the xparity symmetry.
        Returns +1 or -1 as an integer.
        """
        return self._sector

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
        msc = msc.copy()

        # delete elements which do not commute with symmetry operator
        keep = parity(msc['signs']) == 0
        conserved = np.all(keep)
        msc = msc[keep]

        terms_to_mod = np.nonzero(msc['masks'] >> (self.L-1))
        msc['masks'][terms_to_mod] ^= (1 << self.L) - 1

        if self.sector == -1:
            msc['coeffs'][terms_to_mod] *= -1

        msc = combine_and_sort(msc)

        if check_conserves:
            return msc, conserved
        else:
            return msc

    def convert_state(self, state):
        """
        Convert a state on the XParity subspace to one on its parent, or
        vice versa.

        Parameters
        ----------

        state : State
            The input state

        Returns
        -------

        State
            The converted state
        """
        state.assert_initialized()

        istart, iend = state.vec.getOwnershipRange()
        n_in = len(state)

        block_size = 1024

        # convert to parent
        if state.subspace is self:
            rtn_state = states.State(subspace=self.parent)

            flip_mask = (1 << self.L) - 1
            for block_start in range(istart, iend, block_size):
                block_end = min(iend, block_start + block_size)

                # get the indices where values should be set
                from_idxs = np.arange(block_start, block_end, dtype=dnm_int_t)
                from_states = self.idx_to_state(from_idxs)
                to_idxs = self.parent.state_to_idx(flip_mask ^ from_states)
                rtn_state.vec[to_idxs] = state.vec[from_idxs]

            rtn_state.vec.assemble()

            if self.sector == -1:
                # flip sector of second half of vector for - subspace
                rtn_state.vec.scale(-1)

            # first half is easier---indices are the same!
            rtn_state.vec[istart:iend] = state.vec[istart:iend]

        # convert from parent
        elif state.subspace is self.parent:
            rtn_state = states.State(subspace=self)

            # second half of vector
            start = max(n_in//2, istart)
            end = iend
            if start < end:
                flip_mask = (1 << self.L) - 1
                for block_start in range(start, end, block_size):
                    block_end = min(end, block_start + block_size)

                    # get the indices where values should be set
                    from_idxs = np.arange(block_start, block_end, dtype=dnm_int_t)
                    from_states = self.parent.idx_to_state(from_idxs)
                    to_idxs = self.state_to_idx(flip_mask ^ from_states)
                    rtn_state.vec[to_idxs] = state.vec[from_idxs]

            rtn_state.vec.assemble()
            if self.sector == -1:
                # flip sector of second half of vector for - subspace
                rtn_state.vec.scale(-1)

            if istart < n_in//2:
                start = istart
                end = min(n_in//2, iend)
                rtn_state.vec.setValues(np.arange(start, end, dtype=dnm_int_t),
                                        state.vec[start:end],
                                        addv=True)

        else:
            raise ValueError('subspace of input state must be this XParity subspace '
                             'or its parent')

        rtn_state.vec.assemble()
        rtn_state.vec.scale(1/np.sqrt(2))  # normalize

        rtn_state.set_initialized()

        return rtn_state

    def __hash__(self):
        return hash(('XParity', self.sector, self.parent))

    def __repr__(self):
        return f'XParity({repr(self.parent)}, sector={self.sector:+d})'

    @property
    def _L(self):
        return self.parent.L

    @_L.setter
    def _L(self, value):
        self.parent.L = value

    def _get_dimension(self):
        return self.parent.get_dimension()//2

    def _idx_to_state(self, idx):
        # representative states are the first N/2
        # so we can just call parent function directly
        return self.parent.idx_to_state(idx)

    def _state_to_idx(self, state):
        # this function takes a representation of a "representative"
        # state---so it must start with 0 (have spin L-1 in state 0)
        if np.count_nonzero(state >> (self.L-1)):
            raise ValueError('invalid state')

        return self.parent.state_to_idx(state)

    def _to_c(self):
        return self.parent._to_c()
