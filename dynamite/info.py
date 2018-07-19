
from . import config
from sys import stderr

def write(level,msg):
    """
    Write information to stderr, if ``config.info_level`` is set at or higher
    than ``level`` argument.
    """
    if level > config.info_level:
        return

    if config.initialized:
        from petsc4py import PETSc
        process_str = ', proc. %d' % PETSc.COMM_WORLD.rank
    else:
        process_str = ''

    print(('[info %s%s]: ' % (str(level), process_str))+msg,file=stderr)
