
import numpy as np
from copy import deepcopy

from . import validate

# TODO: this will be serial_backend eventually
from .backend import backend

# TODO: implement checks that subspace is valid?

def class_to_enum(subspace_type):
    to_enum = {
        Parity : backend.SubspaceType.PARITY,
        Full : backend.SubspaceType.FULL
    }
    return to_enum[subspace_type]

class Subspace:
    # base class for all subspaces. Each one should define these
    # member functions.

    _space = None

    def __init__(self,space=None,L=None):
        self.space = space
        self.L = L # should use config value

    def __eq__(self,s):
        if not isinstance(s,Subspace):
            raise ValueError('Cannot compare Subspace to non-Subspace type')

        return type(s) == type(self) and s.space == self.space

    @property
    def space(self):
        """
        A parameter defining the subspace---for example, for parity,
        it might be even or odd.
        """
        return self._space

    # need to be careful here--if we change the subspace but the matrix
    # isn't destroyed, it could mess everything up!
    @space.setter
    def space(self):
        raise NotImplementedError()

    def get_size(self,L=None):
        """
        Get the dimension of the Hilbert space for a spin chain length `L`.

        Parameters
        ----------

        L : int, optional
            The spin chain length. Can be omitted if Subspace.L is set.
        """

        if L is None:
            L = self.L

        L = validate.L(L)

        return self._get_size(L)

    @classmethod
    def _get_size(cls,L):
        raise NotImplementedError()

    def idx_to_state(self,idx):
        """
        Maps a matrix or vector index to an integer representing the
        corresponding spin configuration (or, equivalently, an index
        in the full Hilbert space)
        """
        raise NotImplementedError()

    def state_to_idx(self,state):
        """
        The inverse mapping of :meth:`idx_to_state`.
        """
        raise NotImplementedError()

    def update_operator(self,which,operator):
        """
        Updates an operator when the subspace changes. This is allowed to be
        overridden by particular subspaces, so that clever things can be done to save
        memory.
        """
        if which == 'left':
            if operator.left_subspace != self:
                operator.destroy_mat()
        elif which == 'right':
            if operator.right_subspace != self:
                operator.destroy_mat()
        else:
            raise ValueError('which must be "left" or "right"')

    def copy(self):
        return deepcopy(self)


class Parity(Subspace):

    @classmethod
    def _get_size(cls,L):
        return 1<<(L-1)

    @Subspace.space.setter
    def space(self,value):
        if value in [0,'even']:
            self._space = 0
        elif value in [1,'odd']:
            self._space = 1
        elif value is None:
            self._space = None
        else:
            raise ValueError('Invalid parity space "'+str(value)+'" (valid choices are 0,1,"even","odd",None)')

    @classmethod
    def parity(cls,x):
        x = np.array(x)
        p = x&1
        x >>= 1
        while np.any(x):
            p ^= x&1
            x >>= 1
        return p

    def idx_to_state(self,idx):

        # TODO: need to be careful about memory usage if this is called with a huge array

        if self.space is None:
            raise ValueError('Must set parity space (even or odd) before calling idx_to_state.')

        if self.L is None:
            raise ValueError('Must set spin chain size for parity object (Parity.L) before calling '
                             'idx_to_state.')

        p = self.parity(idx)
        prefix = p^self.space
        return idx | (prefix << (self.L-1))

    def state_to_idx(self,state):
        if self.space is None:
            raise ValueError('Must set parity space (even or odd) before calling state_to_idx.')

        if self.L is None:
            raise ValueError('Must set spin chain size for parity object (Parity.L) before calling '
                             'state_to_idx.')

        idxs = state & (~((-1)<<(self.L-1)))
        bad = self.parity(state) != self.space
        idxs[bad] = -1
        return idxs

class Full(Subspace):

    @classmethod
    def _get_size(cls,L):
        return 1<<L

    @Subspace.space.setter
    def space(self,value):
        if not (value is None or value == 0):
            raise ValueError('Only valid choice for full space subspace specifier is "None" or 0.')
        self._space = 0

    def state_to_idx(self,state):
        if not 0 <= state < self.get_size():
            raise ValueError('State %d out of range (%d,%d)' % (state,0,self.get_size()))
        return state

    def idx_to_state(self,idx):
        if not 0 <= idx < self.get_size():
            raise ValueError('State %d out of range (%d,%d)' % (idx,0,self.get_size()))
        return idx
