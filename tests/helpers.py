
import numpy as np
from scipy.linalg import expm
from dynamite import config
from dynamite.tools import build_state,vectonumpy
from dynamite.operators import Sigmax,Sigmay,Sigmaz
from petsc4py.PETSc import COMM_WORLD,NormType

from numpy_operators import *

try:
    import qutip as qtp
except ImportError:
    qtp = None

CW = COMM_WORLD.tompi4py()
PROC_0 = CW.Get_rank() == 0

def dnm_to_np(H):
    dim = 2**H.L

    if PROC_0:
        ret = np.ndarray((dim,dim),dtype=np.complex128)

    s1 = build_state(H.L)
    s2 = build_state(H.L)

    s1.set(0)
    for i in range(dim):
        if i > 0:
            s1.setValue(i-1,0)
        s1.setValue(i,1)
        s1.assemblyBegin()
        s1.assemblyEnd()
        H.get_mat().mult(s1,s2)
        r = vectonumpy(s2)

        if PROC_0:
            ret[:,i] = r

    if PROC_0:
        return ret
    else:
        return None

def dnm_np_operator(i,index,L):
    if i == 'x':
        d = Sigmax(index,L=L)
        n = np_sigmax(index,L)
    elif i == 'y':
        d = Sigmay(index,L=L)
        n = np_sigmay(index,L)
    elif i == 'z':
        d = Sigmaz(index,L=L)
        n = np_sigmaz(index,L)
    else:
        raise ValueError('Type \'%s\' is not valid.' % i)
    return (d,n)

def compare_and_scatter(c,*args):
    '''
    Check that value is true on process 0.
    Then, broadcast that truth to all other
    processes and check it there too (so that
    we don't end up hanging if process 0 fails
    a test).

    c is a function that returns True or False,
        with the arguments *args.
    '''
    if PROC_0:
        r,msg = c(*args)
    else:
        r = None
        msg = ''

    return CW.bcast(r,root=0),msg

def check_dnm_np(d,n):
    d_np = dnm_to_np(d)
    d_norm = d.get_mat().norm(NormType.INFINITY)

    return compare_and_scatter(_matrix_checks,n,d,d_np,d_norm)

def check_vecs(d,n,max_nrm=0):
    d_np = vectonumpy(d)
    return compare_and_scatter(_vector_checks,d_np,n,max_nrm)

def check_evolve(d,n,state,t=0.1):
    ds = build_state(L=d.L,state=state)
    dr = d.evolve(ds,t=t)

    ns = vectonumpy(ds)

    if PROC_0:
        nr = expm(-1j*t*n).dot(ns)
    else:
        nr = None

    return check_vecs(dr,nr,max_nrm=1E-15)

def check_close(a,b,tol=1E-12):
    return compare_and_scatter(_close,a,b,tol)

def _close(a,b,tol):
    msg = ''
    r = abs(a-b) < tol
    if not r:
        msg += '%f and %f not within tolerance %f' % (a,b,tol)
    return r,msg

def _vector_checks(d,n,max_nrm):
    '''
    do the following checks on vectors:

      -- if max_nrm is zero, they should be exactly identical
      -- otherwise, take the difference and then check that
         the norm < max_nrm
    '''
    msg = ''
    if max_nrm > 0:
        diff = d-n
        nrm = np.linalg.norm(diff)
        r = nrm < max_nrm
        if not r:
            msg += 'vector difference norm '+str(nrm)
            msg += ' not smaller than tolerance '+str(max_nrm)
    else:
        r = np.all(d==n)
        if not r:
            msg += 'vectors not equal.'
            if len(n) <= 64:
                msg += '\ndynamite vector:\n'+str(d)
                msg += '\nnumpy vector:\n'+str(n)

    return r,msg

def _matrix_checks(n,d,dnm_np,dnm_norm):
    '''
    make the following checks:
      -- equality
      -- nnz
      -- norm
      -- global shell
      -- qutip, if available

    note: this function will ONLY run on process 0
    So, don't do anything that needs to run in parallel!
    '''

    r = True
    msg = ''

    ### equality

    tmp = np.allclose(n,dnm_np)
    if not tmp:
        msg += '''matrices not equal.
dynamite matrix: %s
numpy matrix: %s
''' % (str(dnm_np), str(n))

        # print some nonzero elements of difference
        msg += 'nonzero elements of difference:\n'
        diff = dnm_np-n
        nz = np.transpose(np.nonzero(diff))
        for i,(idx,idy) in enumerate(nz):
            # don't blow up the output if there are a lot
            if i > 20:
                break
            msg += '%d %d %s\n' % (idx,idy,diff[idx,idy])

    r = r and tmp

    ### nnz

    tmp = np.all(d.nnz==np.max(np.count_nonzero(n,axis=0)))
    if not tmp:
        msg += ('nnz not equal. dynamite nnz=%d, numpy nnz=%d\n'
                % (d.nnz,np.max(np.count_nonzero(n,axis=0))))

    r = r and tmp

    ### infinity norm

    np_norm = np.linalg.norm(n,ord=np.inf)
    tmp = np.isclose(np_norm,dnm_norm)
    if not tmp:
        msg += ('norm not equal. dynamite norm=%f, numpy norm=%f\n'
                % (dnm_norm,np_norm))

    r = r and tmp

    ### global shell

    tmp = d.use_shell == config.global_shell
    if not tmp:
        msg += ('use_shell value %s not equal to global shell value %s\n'
                % (str(d.use_shell),str(config.global_shell)))

    r = r and tmp

    ### qutip

    if qtp is not None:
        qtp_o = d.build_qutip().full()
        tmp = np.allclose(qtp_o,dnm_np)
        if not tmp:
            msg += '''qutip matrix not equal to dynamite matrix!
qutip matrix: %s
dynamite matrix: %s
''' % (str(qtp_o),str(dnm_np))

        r = r and tmp

    return r,msg
