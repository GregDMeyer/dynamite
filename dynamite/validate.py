
import numpy as np
from .msc_tools import msc_dtype

def L(x):
    try:
        if int(x) != x or x < 0:
            raise ValueError()
    except:
        raise ValueError('Spin chain length L must be a nonnegative integer (got %s)' % str(x))

    return x

def subspace(s):
    from .subspaces import Subspace
    if not isinstance(s,Subspace):
        raise ValueError('subspace can only be set to objects of Subspace type, or None')
    return s

def msc(x):
    x = np.array(x, copy=False, dtype=msc_dtype)
    return x

def brackets(x):
    if x not in ['()', '[]', '']:
        raise ValueError("Brackets must be one of '()', '[]', or ''")
    return x

def shell(s):
    valid_options = [False, 'cpu', 'gpu']
    if s not in valid_options:
        raise ValueError('Options for shell matrices are %s.' % \
                         ', '.join(str(x) for x in valid_options))
    return s

def info_level(level):
    valid_levels = [0,1,2]
    if level not in valid_levels:
        raise ValueError('invalid info level. options are %s' % str(valid_levels))
