
from . import config, validate, subspaces
from .tools import complex_enabled

import numpy as np
from os import urandom
from time import time

class State:
    """
    Class representing a state.

    Parameters
    ----------

    L : int
        Spin chain length. Can be ommitted if config.L is set.

    subspace : dynamite.subspace.Subspace, optional
        The subspace on which the state should be defined.
        If not specified, defaults to config.subspace.

    state : int or str, optional
        An initial product state to set the state to. Also accepts ``'random'``. The
        state can also be initialized later with the :meth:`set_product` and
        :meth:`set_random` methods.

    seed : int, optional
        If the ``state`` argument is set to ``'random'``, the seed for the random number
        generator. This argument is ignored otherwise.
    """

    def __init__(self, L = None, subspace = None, state = None, seed = None):

        config._initialize()
        from petsc4py import PETSc

        if L is None:
            L = config.L

        self.L = validate.L(L)

        if subspace is None:
            if config.subspace is not None:
                subspace = config.subspace
            else:
                subspace = subspaces.Full()

        self._subspace = validate.subspace(subspace)
        self._subspace.L = self.L

        self._vec = PETSc.Vec().create()
        self.vec.setSizes(subspace.get_dimension())
        self.vec.setFromOptions()

        if state is not None:
            if state == 'random':
                self.set_random(seed=seed)
            else:
                self.set_product(state)

    def copy(self, result=None):
        if result is None:
            result = State(self.L, self.subspace.copy())

        if self.subspace != result.subspace:
            raise ValueError('subspace of state and result must match')

        self.vec.copy(result.vec)
        return result

    @property
    def subspace(self):
        """
        The space on which the vector is defined.

        See :mod:`dynamite.subspaces` for details.
        """
        return self._subspace

    @property
    def vec(self):
        """
        The PETSc vector containing the state data.

        petsc4py Vec methods can be used through this interface--for example, to find the norm of a
        State `s`, one can do `state.vec.norm()`. The methods don't seem to be documented anywhere,
        but are fairly transparent from looking at the petsc4py source code.
        """
        return self._vec

    @classmethod
    def str_to_state(cls, s, L):
        '''
        Convert a string to an integer whose bitwise representation is the spin
        configuration (0=↑, 1=↓) of a product state. The characters
        'D' and 'U' represent down and up spins, like ``"DUDDU...UDU"`` (D=↓, U=↑).

        .. note::
            For the string representation, the leftmost character is spin index 0. For
            an integer representation, the rightmost (least significant) bit is!

        Parameters
        ----------
        s : int or string
            The state. If an integer is passed, the same integer is returned.

        L : int
            The length of the spin chain

        Returns
        -------
        int
            The state
        '''

        if isinstance(s, str):
            if len(s) != L:
                raise ValueError('state string must have length L')

            if not all(c in ['U','D'] for c in str(s)):
                raise ValueError('only character U and D allowed in state')

            state = 0
            for i,c in enumerate(s):
                if c == 'D':
                    state += 1<<i

        else:
            state = int(s)

        return state

    def set_product(self, s):
        """
        Initialize to a product state. Can be specified either be an integer whose binary
        representation represents the spin configuration (0=↑, 1=↓) of a product state, or a string
        of the form ``"DUDDU...UDU"`` (D=↓, U=↑). If it is a string, the string's length must
        equal ``L``.

        .. note:
            In integer representation, the least significant bit represents spin 0. So, if you look
            at a binary representation of the integer (for example with Python's `bin` function)
            spin 0 will be the rightmost bit!

        Parameters
        ----------

        s : int or str
            A representation of the state.
        """

        s2i_result = self.subspace.state_to_idx(self.str_to_state(s, self.L))
        if isinstance(s2i_result, tuple):
            idx, sign = s2i_result
        else:
            idx = s2i_result

        if idx == -1:
            raise ValueError('Provided initial state not in requested subspace.')

        self.vec.set(0)

        istart, iend = self.vec.getOwnershipRange()
        if istart <= idx < iend:
            self.vec[idx] = 1

        self.vec.assemblyBegin()
        self.vec.assemblyEnd()

    @classmethod
    def generate_time_seed(cls):

        config._initialize()
        from petsc4py import PETSc

        CW = PETSc.COMM_WORLD.tompi4py()

        if CW.rank == 0:
            seed = int(time())
        else:
            seed = None

        return CW.bcast(seed, root = 0)

    def set_random(self, seed = None, normalize = True):
        """
        Initialized to a normalized random state.

        .. note::
            When running under MPI with multiple processes, the seed is incremented
            by the MPI rank, so that each process generates different random values.

        Parameters
        ----------

        seed : int, optional
            A seed for numpy's PRNG that is used to build the random state. The user
            should pass the same value on every process.

        normalize : bool
            Whether to rescale the random state to have norm 1.
        """

        config._initialize()
        from petsc4py import PETSc

        istart, iend = self.vec.getOwnershipRange()

        R = np.random.RandomState()

        if seed is None:
            try:
                seed = int.from_bytes(urandom(4), 'big', signed=False)
            except NotImplementedError:
                seed = self.generate_time_seed()

        # if my code is still being used in year 2038, wouldn't want it to
        # overflow numpy's PRNG seed range ;)
        R.seed((seed + PETSc.COMM_WORLD.rank) % 2**32)

        local_size = iend-istart

        if complex_enabled():
            self.vec[istart:iend] =    R.standard_normal(local_size) + \
                                    1j*R.standard_normal(local_size)
        else:
            self.vec[istart:iend] = R.standard_normal(local_size)

        self.vec.assemblyBegin()
        self.vec.assemblyEnd()

        if normalize:
            self.vec.normalize()

    @classmethod
    def _to_numpy(cls, vec, to_all = False):
        '''
        Collect PETSc vector (split across processes) to a
        numpy array on process 0, or to all processes if
        `to_all == True`.

        Parameters
        ----------
        vec : petsc4py.PETSc.Vec
            The input vector

        to_all : bool, optional
            Whether to create numpy vectors on all processes, or
            just on process 0.

        Returns
        -------
        numpy.ndarray or None
            A numpy array of the vector, or ``None``
            on all processes other than 0 if `to_all == False`.
        '''

        from petsc4py import PETSc

        # scatter seems to be broken for CUDA vectors
        if PETSc.COMM_WORLD.size > 1:
            # collect to process 0
            if to_all:
                sc, v0 = PETSc.Scatter.toAll(vec)
            else:
                sc, v0 = PETSc.Scatter.toZero(vec)

            sc.begin(vec, v0)
            sc.end(vec, v0)

            if not to_all and PETSc.COMM_WORLD.rank != 0:
                return None
        else:
            v0 = vec

        ret = np.ndarray((v0.getSize(),), dtype=np.complex128)
        ret[:] = v0[:]

        return ret

    def to_numpy(self, to_all = False):
        """
        Return a numpy representation of the state.

        Parameters
        ----------
        to_all : bool
            Whether to return the vector on all MPI ranks (True),
            or just rank 0 (False).
        """
        return self._to_numpy(self.vec, to_all)

    def dot(self, x):
        return self.vec.dot(x.vec)

    def __imul__(self, x):
        self.vec.scale(x)
        return self

    def __itruediv__(self, x):
        self.vec.scale(1/x)
        return self

    def __len__(self):
        return self.vec.getSize()


auto_wrap = ['norm', 'normalize']
for petsc_func in auto_wrap:
    setattr(State, petsc_func, lambda self, f=petsc_func: getattr(self.vec, f)())
