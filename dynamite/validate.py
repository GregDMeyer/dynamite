
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
