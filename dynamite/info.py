
from . import config
from sys import stderr

valid_levels = [0,1]

def validate_level(level):
    if level not in valid_levels:
        raise ValueError('invalid info level. options are %s' % str(valid_levels))

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
