
import slepc4py
from .subspace import check_subspace,Full

# handle global configuration

class _Config:
    """
    Package-wide configuration of dynamite.
    """

    initialized = False
    _L = None
    _shell = False
    _subspace = Full()

    def initialize(self,slepc_args=None):
        """
        Initialize PETSc/SLEPc with various arguments (which would be
        passed on the command line for a C program).

        Only the first call to this function has any effect, and this
        function is automatically called with no arguments when any
        dynamite submodule is imported. Thus, one must call it before
        importing any submodules.

        Parameters
        ==========

        slepc_args : list of str
            The arguments to SLEPc initialization.
        """
        if slepc_args is None:
            slepc_args = []

        if not self.initialized:
            slepc4py.init(slepc_args)
            self.initialized = True
        else:
            if slepc_args:
                raise RuntimeError('initialize has already been called. Perhaps '
                                   'you already imported a dynamite submodule?')

    @property
    def global_L(self):
        raise TypeError('config.global_L has been changed to config.L, please use that instead.')

    @global_L.setter
    def global_L(self,value):
        raise TypeError('config.global_L has been changed to config.L, please use that instead.')

    @property
    def global_shell(self):
        raise TypeError('config.global_shell has been changed to config.shell, please use that instead.')

    @global_shell.setter
    def global_shell(self,value):
        raise TypeError('config.global_shell has been changed to config.shell, please use that instead.')

    @property
    def L(self):
        """
        A global spin chain length that will be applied to all matrices and states,
        unless they are explicitly set to a different size. Is **not** retroactive---
        will not set the size for any objects that have already been created.
        """
        return self._L

    @L.setter
    def L(self,value):

        if value is None:
            self._L = value
            return

        L = int(value)
        if L != value:
            raise ValueError('L must be an integer or None.')
        if L < 1:
            raise ValueError('L must be >= 1.')

        self._L = L

    @property
    def shell(self):
        """
        Whether to use shell matrices everywhere (True), or to use standard
        PETSc matrices (False, default). Experimental support for GPU shell matrices ('gpu')
        is also included if the package could find a CUDA compiler during build.
        """
        return self._shell

    @shell.setter
    def shell(self,value):

        if value not in [True,False,'gpu']:
            raise ValueError('invalid value for config.global_shell')

        if value == 'gpu':

            if not self.initialized:
                raise RuntimeError('Must call config.initialize() before setting '
                                   'global_shell to "gpu".')

            from .backend.backend import have_gpu_shell

            # maybe should do this check at build time, not now?
            if not have_gpu_shell():
                raise RuntimeError('GPU shell matrices not enabled (could not find nvcc '
                                   'during build)')

        self._shell = value

    @property
    def subspace(self):
        """
        The subspace to use for all operators and states. Can also be set for individual
        operators and states--see :meth:`dynamite.operators.Operator.subspace` for details.
        """
        return self._subspace

    @subspace.setter
    def subspace(self,value):
        check_subspace(value)
        self._subspace = value

config = _Config()
