
from . import config, validate
from .tools import vectonumpy

import numpy as np
from os import urandom
from time import time

# TODO: wrap some of the most common functions of PETSc vectors -- like norm, binary operators, etc.
# could probably do it in some clever way

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

    def __init__(self,L=None,subspace=None,state=None,seed=None):

        config.initialize()
        from petsc4py import PETSc

        if L is None:
            L = config.L

        self.L = validate.L(L)
        self._subspace = None
        self._vec = None

        if subspace is None:
            subspace = config.subspace

        self._subspace = validate.subspace(subspace)
        self._subspace.L = self.L

        self._vec = PETSc.Vec().create()
        self._vec.setSizes(subspace.get_size(L))
        self._vec.setFromOptions()

        if state is not None:
            if state == 'random':
                self.set_random(seed=seed)
            else:
                self.set_product(state)

    @property
    def subspace(self):
        """
        The space on which the vector is defined.

        See :module:`dynamite.subspace` for details.
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

    def set_product(self,s):
        """
        Initialize to a product state. Can be specified either be an integer whose binary
        representation represents the spin configuration (0=↓, 1=↑) of a product state, or a string
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

        if isinstance(s,str):
            state = 0
            if len(s) != self.L:
                raise ValueError('state string must have length L')
            if not all(c in ['U','D'] for c in s):
                raise ValueError('only character U and D allowed in state')
            for i,c in enumerate(s):
                if c == 'U':
                    state += 1<<i

        elif isinstance(s,int):
            state = s

        else:
            raise TypeError('State must be an int or str.')

        idx = self.subspace.state_to_idx(state)

        self.vec.set(0)
        self.vec[idx] = 1

        self.vec.assemblyBegin()
        self.vec.assemblyEnd()

    def set_random(self,seed=None):
        """
        Initialized to a normalized random state.

        Parameters
        ----------

        seed : int, optional
            A seed for numpy's PRNG that is used to build the random state.
        """

        config.initialize()
        from petsc4py import PETSc

        istart,iend = self.vec.getOwnershipRange()

        R = np.random.RandomState()

        if seed is None:
            try:
                seed = int.from_bytes(urandom(4),'big',signed=False)
            except NotImplementedError:
                # synchronize the threads
                PETSc.COMM_WORLD.barrier()
                seed = int(time())

        # if my code is still being used in year 2106, wouldn't want it to
        # overflow numpy's PRNG seed range ;)

        # TODO: just use bcast instead of this craziness
        # I also want to make sure that if the time was slightly different on
        # different processes that they don't end up with the same seed
        R.seed(seed + (63 * PETSc.COMM_WORLD.rank) % 2**32)

        local_size = iend-istart
        self.vec[istart:iend] =    R.standard_normal(local_size) + \
                                1j*R.standard_normal(local_size)

        self.vec.assemblyBegin()
        self.vec.assemblyEnd()
        self.vec.normalize()

    def to_numpy(self):
        """
        Return a numpy representation of the state.
        """
        return vectonumpy(self.vec)
