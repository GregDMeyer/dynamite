
import numpy as np

def L(x):
    try:
        if int(x) != x or x < 1:
            raise ValueError()
    except:
        raise ValueError('Spin chain length L must be a positive integer.')

    return x

def subspace(s):
    from .subspace import Subspace
    if not isinstance(s,Subspace):
        raise ValueError('subspace can only be set to objects of Subspace type, or None')
    return s

def MSC(x):
    from ._imports import get_import
    backend = get_import('backend')
    x = np.array(x, copy=False, dtype=backend.MSC_dtype)
    return x

def brackets(x):
    if x not in ['()', '[]', '']:
        raise ValueError("Brackets must be one of '()', '[]', or ''")
    return x

def shell(s):
    if s not in [True, False, 'gpu']:
        raise ValueError('Options for shell matrices are True, False, or "gpu".')
    return s
