
from os import environ
import slepc4py
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
    _info_level = 0
    _subspace = None
    _gpu = False

    def initialize(self, slepc_args=None, version_check=True, gpu=False):
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

        if slepc_args is None:
            slepc_args = []

        if gpu:
            slepc_args += [
                '-vec_type', 'cuda',
                '-mat_type', 'aijcusparse',
            ]

        # prevent PETSc from being sad if we don't use gpu aware mpi
        if not self.initialized and bbuild.have_gpu_shell():
            slepc_args += ['-use_gpu_aware_mpi', '0'] # we only use one process anyway

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

        slepc4py.init(slepc_args)
        self.initialized = True
        self._gpu = gpu

        from petsc4py import PETSc
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
        operators and states--see :attr:`dynamite.operators.Operator.subspace` for details.
        """
        return self._subspace

    @subspace.setter
    def subspace(self,value):
        self._subspace = validate.subspace(value)

    @property
    def gpu(self):
        """
        Whether to run the computations on a GPU. This property is read-only. To use
        GPUs, :meth:`initialize()` must be called with ``gpu=True``.
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
    except (FileExistsError, PermissionError):
        return

    # finally do the check

    branch = bbuild.get_build_branch()
    url = 'https://api.github.com/repos/GregDMeyer/dynamite/git/refs/heads/{branch}'
    url = url.format(branch=branch)

    try:
        with request.urlopen(url, timeout=1) as url_req:
            try:
                data = json.load(url_req)
            except TypeError: # python < 3.6
                data = json.loads(url_req.readall().decode('utf-8'))

            commit = data['object']['sha']

        build_commit = bbuild.get_build_version()
        if not commit.startswith(build_commit):
            if 'DNM_DOCKER' in environ:
                update_msg = 'Please pull the latest image from DockerHub.'
            else:
                update_msg = 'Please update!'

            print('Changes have been pushed to GitHub since dynamite was '
                  'installed.\n' + update_msg, file=stderr)

    # in general, catching all exceptions is a bad idea. but here, no matter
    # what happens we just want to give up on the check
    except:
        pass
