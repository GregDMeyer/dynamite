
import numpy as np

def np_tensor_prod(a,b):
    return np.hstack(np.hstack(np.tensordot(a,b,axes=0)))

def np_identity_product(a, index, L):
    ret = None
    for i in range(L):
        if i == index:
            this_op = a
        else:
            this_op = np.identity(2)
        if ret is None:
            ret = this_op
        else:
            ret = np_tensor_prod(this_op,ret)
    return ret

def np_sigmax(index,L):
    sx = np.array([[0,1],[1,0]],dtype=np.complex128)
    return np_identity_product(sx,index,L)

def np_sigmay(index,L):
    sy = np.array([[0,-1j],[1j,0]],dtype=np.complex128)
    return np_identity_product(sy,index,L)

def np_sigmaz(index,L):
    sz = np.array([[1,0],[0,-1]],dtype=np.complex128)
    return np_identity_product(sz,index,L)

def np_sigmai(i,*args,**kwargs):
    if i == 'x':
        return np_sigmax(*args,**kwargs)
    elif i == 'y':
        return np_sigmay(*args,**kwargs)
    elif i == 'z':
        return np_sigmaz(*args,**kwargs)
    else:
        raise ValueError('Type \'%s\' is not valid.' % i)