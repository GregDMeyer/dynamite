
import numpy as np
from .msc_tools import msc_dtype


def L(x):
    try:
        if int(x) != x or x < 0:
            raise ValueError()
    except:
        raise ValueError('Spin chain length L must be a nonnegative integer (got %s)' % str(x))

    if x > 63:
        raise ValueError('Spin chain lengths greater than 63 not supported.')

    int_t = msc_dtype['masks']
    if int_t == np.int32 and x > 31:
        raise ValueError('Spin chain lengths greater than 31 not supported when '
                         'using 32 bit integers. Rebuild PETSc with the option '
                         '"--with-64-bit-indices".')

    return x


def subspace(s):
    from .subspaces import Subspace
    if not isinstance(s, Subspace):
        raise ValueError('subspace can only be set to objects of Subspace type')
    return s


def msc(x):
    x = np.array(x, copy=False, dtype=msc_dtype)
    return x


def shell(s):
    if not isinstance(s, bool):
        raise ValueError("Shell must be set to True or False. To use GPU shell matrices, "
                         "just do config.initialize(gpu=True), and set shell to True.")
    return s
