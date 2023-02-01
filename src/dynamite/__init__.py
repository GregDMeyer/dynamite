
from os import environ
import slepc4py
from threadpoolctl import threadpool_limits
from . import validate
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
    _subspace = None
    _gpu = False

    def initialize(self, slepc_args=None, version_check=True, gpu=None):
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

        gpu : bool
            Whether to run all computations using a GPU instead of the CPU.
        """
        if self.initialized:
            raise RuntimeError('dynamite.config.initialize() can only be called once.')

        self._initialize(slepc_args, version_check, gpu)

    def _initialize(self, slepc_args=None, version_check=True, gpu=None):
        """
        This function should only be called by internal dynamite code, to initialize
        things if user didn't manually call initialize()
        """

        if self.initialized:
            return

        if slepc_args is None:
            slepc_args = []

        if gpu is None:
            gpu = bbuild.have_gpu_shell()

        if gpu:
            if not bbuild.have_gpu_shell():
                raise RuntimeError('Cannot initialize for GPU; this build of '
                                   'dynamite/petsc was not configured with '
                                   'GPU functionality')

            slepc_args += [
                '-vec_type', 'cuda',
                '-mat_type', 'aijcusparse',
            ]

        slepc_args += [
            # to avoid an extra useless file being created when we save
            # State objects
            '-viewer_binary_skip_info',

            # so it doesn't warn us if we don't use all these options
            '-options_left', '0'
        ]

        # prevent PETSc from being sad if we don't use gpu aware mpi
        if not self.initialized and bbuild.have_gpu_shell():
            slepc_args += ['-use_gpu_aware_mpi', '0'] # we only use one process anyway

        if bbuild.petsc_initialized():
            raise RuntimeError('PETSc has been initialized but dynamite has not. '
                               'Call dynamite.config.initialize(args) before importing '
                               'any PETSc modules or interfacing with PETSc functionality '
                               '(like building matrices).')

        slepc4py.init(slepc_args)
        self.initialized = True
        self._gpu = gpu

        from petsc4py import PETSc

        # disable extra thread-level parallelism that can interfere with MPI
        # parallelism
        if PETSc.COMM_WORLD.size != 1:
            threadpool_limits(limits=1)

        if version_check and PETSc.COMM_WORLD.rank == 0:
            check_version()

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
        Whether to use standard PETSc matrices (``False``, default), or shell
        matrices (``True``).
        """
        return self._shell

    @shell.setter
    def shell(self,value):
        self._shell = validate.shell(value)

    @property
    def subspace(self):
        """
        The subspace to use for all operators and states. Can also be set for individual
        operators and states--see :attr:`dynamite.operators.Operator.subspace` for details.
        """
        return self._subspace

    @subspace.setter
    def subspace(self, value):
        if value is None:
            self._subspace = None
        else:
            self._subspace = validate.subspace(value)

    @property
    def gpu(self):
        """
        Whether to run the computations on a GPU. This property is read-only. To use
        GPUs, :meth:`initialize()` must be called with ``gpu=True`` (default when
        built with GPU support).
        """
        return self._gpu

config = _Config()

def check_version():
    """
    Check for any updates to dynamite, skipping if a check has been performed
    in the last day.
    """

    from urllib import request
    import json
    from os import remove
    from os.path import isfile
    from time import time
    from sys import stderr

    # only check once a day for a new version so that we don't DOS GitHub
    # we save a file with the time of the last check in it
    filename = '.dynamite'
    if isfile(filename):
        with open(filename) as f:
            last_check = float(f.read().strip())
    else:
        last_check = 0

    # if less than a day has elapsed, return
    cur_time = time()
    one_day = 24*60*60
    if cur_time - last_check < one_day:
        return

    # otherwise, we should write out a new time file
    try:
        with open(filename+'_lock', 'x'):
            with open(filename, 'w') as f:
                f.write(str(time()))
        remove(filename+'_lock')

    # another process is doing this at the same time,
    # or we don't have write permission here
    except (FileExistsError, PermissionError, OSError):
        return

    # finally do the check

    url = 'https://api.github.com/repos/GregDMeyer/dynamite/releases/latest'
    try:
        with request.urlopen(url, timeout=1) as url_req:
            data = json.load(url_req)

    # in general, catching all exceptions is a bad idea. but here, no matter
    # what happens we just want to give up on the check
    except:
        return

    release_version = data["tag_name"][1:]  # tag_name starts with 'v'
    if release_version != bbuild.get_build_version():
        print('A new version of dynamite has been released!', file=stderr)

        if 'DNM_DOCKER' in environ:
            update_msg = 'Please pull the latest image from DockerHub.'
        else:
            update_msg = 'Please update!'

        print(update_msg, file=stderr)
