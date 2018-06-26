
import slepc4py
from . import validate
from .subspace import Full
from ._backend import bbuild

# handle global configuration

class _Config:
    """
    Package-wide configuration of dynamite.
    """

    initialized = False
    mock_backend = False
    _L = None
    _shell = False
    _info_level = 0
    _subspace = Full()

    def initialize(self, slepc_args = None, version_check = True):
        """
        Initialize PETSc/SLEPc with various arguments (which would be
        passed on the command line for a C program).

        Only the first call to this function has any effect. It is automatically
        called when using much of the PETSc/SLEPc functionality (including importing
        `petsc4py.PETSc` or `slepc4py.SLEPc`), so it must be called early (probably
        right after importing dynamite).

        Parameters
        ==========
        slepc_args : list of str
            The arguments to SLEPc initialization.

        version_check : bool
            Whether process 0 should check for a new dynamite version on initialization.
            Can be set to false if the check is unnecessary or causes problems.
        """

        explain_str = 'Call dynamite.config.initialize(args) before importing ' +\
                      'any PETSc modules or interfacing with PETSc functionality ' +\
                      '(like building matrices).'

        if self.initialized:
            if slepc_args:
                raise RuntimeError('dynamite.config.initialize() has already been called. ' +\
                                   explain_str)
            else:
                return

        if bbuild.petsc_initialized():
            raise RuntimeError('PETSc has been initialized but dynamite has not. ' +\
                               explain_str)

        if slepc_args is None:
            slepc_args = []

        slepc4py.init(slepc_args)
        self.initialized = True

        # check that the number of processes is a power of 2 (currently required)
        # TODO: move this check to be near the code that actually requires this
        from petsc4py import PETSc
        mpi_size = int(PETSc.COMM_WORLD.size)  # this cast is needed for MagicMock
        if mpi_size & (mpi_size-1) != 0:
            raise RuntimeError('Number of MPI processes must be a power of 2!')

        if version_check and PETSc.COMM_WORLD.rank == 0:
            from urllib import request
            import json

            branch = bbuild.get_build_branch()
            url = 'https://api.github.com/repos/GregDMeyer/dynamite/git/refs/heads/{branch}'
            url = url.format(branch = branch)

            try:
                with request.urlopen(url, timeout=1) as url_req:
                    try:
                        data = json.load(url_req)
                    except TypeError: # python < 3.6
                        data = json.loads(url_req.readall().decode('utf-8'))

                    commit = data['object']['sha']

                build_commit = bbuild.get_build_version()
                if not commit.startswith(build_commit):
                    print('Changes have been pushed to GitHub since dynamite was installed. '
                          'Please update!')

            # in general, catching all exceptions is a bad idea. but here, no matter
            # what happens we just want to give up on the check
            except:
                pass

    @property
    def global_L(self):
        raise TypeError('config.global_L has been changed to config.L, please use that instead.')

    @global_L.setter
    def global_L(self,value):
        raise TypeError('config.global_L has been changed to config.L, please use that instead.')

    @property
    def global_shell(self):
        raise TypeError('config.global_shell has been changed to config.shell, '
                        'please use that instead.')

    @global_shell.setter
    def global_shell(self,value):
        raise TypeError('config.global_shell has been changed to config.shell, '
                        'please use that instead.')

    @property
    def L(self):
        """
        A global spin chain length that will be applied to all matrices and states,
        unless they are explicitly set to a different size. Is **not** retroactive---
        will not set the size for any objects that have already been created.
        """
        return self._L

    @L.setter
    def L(self, value):
        value = validate.L(value)
        self._L = value

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

        value = validate.shell(value)

        if value == 'gpu':
            if not bbuild.have_gpu_shell():
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
        self._subspace = validate.subspace(value)

    @property
    def info_level(self):
        """
        How verbose to output debug information. Default is 0. Currently, options are:
        0 - no information output about execution
        1 - all debug information printed
        """
        return self._info_level

    @info_level.setter
    def info_level(self,value):
        # need to do this import here, or it will be cyclic
        from info import validate_level
        validate_level(value)
        self._info_level = value

config = _Config()
