
import numpy as np

# TODO: this should probably be included in a member function
def check_subspace(new,old=None):
    if not (isinstance(new,Subspace) or new is None):
        raise ValueError('subspace can only be set to objects of Subspace type, or None')

    # only rebuild if we need to
    if old is None:
        return True
    else:
        return old.needs_rebuild(new)

# TODO: implement checks that subspace is valid?

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

    def needs_rebuild(self,new):
        """
        Whether an operator needs to be rebuilt when the subspace changes to `new`.
        """

        # in general we need to if it's different, this can be overridden by derived classes.
        return self != new


class Parity(Subspace):

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
            raise ValueError('Must set spin chain size for parity object (Parity.L) before calling idx_to_state.')

        p = self.parity(idx)
        prefix = p^self.space
        return idx | (prefix << (self.L-1))

    def state_to_idx(self,state):
        if self.space is None:
            raise ValueError('Must set parity space (even or odd) before calling state_to_idx.')

        if self.L is None:
            raise ValueError('Must set spin chain size for parity object (Parity.L) before calling state_to_idx.')

        idxs = state & (~((-1)<<(self.L-1)))
        bad = self.parity(state) != self.space
        idxs[bad] = -1
        return idxs

class Full(Subspace):
    @Subspace.space.setter
    def space(self,value):
        if not (value is None or value == 0):
            raise ValueError('Only valid choice for full space subspace specifier is "None" or 0.')
        self._space = 0
