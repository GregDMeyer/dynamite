
import numpy as np
from slepc4py import SLEPc
from petsc4py import PETSc

from .utils import mgr

def evolve(x,H=None,t=None,result=None,tol=None,mfn=None):

    mgr.initialize_slepc()

    if result is None:
        result = H.get_mat().createVecs(side='l')

    if H is not None:
        # check if the evolution is trivial. if H*x = 0, then the evolution does nothing and x is unchanged.
        # In this case MFNSolve fails. to avoid that, we check if we have that condition.
        H.get_mat().mult(x,result)
        if result.norm() == 0:
            result = x.copy()
            return result

    if mfn is None:

        mfn = SLEPc.MFN().create()
        mfn.setType('expokit')

        f = mfn.getFN()
        f.setType(SLEPc.FN.Type.EXP)

        mfn.setFromOptions()

        if t is None or H is None:
            raise Exception('Must supply t and H if not supplying mfn to evolve')

    if tol is not None:
        mfn.setTolerances(tol)

    if H is not None:
        mfn.setOperator(H.get_mat())

    if t is not None:
        f = mfn.getFN()
        f.setScale(-1j*t)

    mfn.solve(x,result)

    return result

def eigsolve(H,getvecs=False,nev=1,target=None,which=None):

    if which is None:
        if target is not None:
            which = 'target'
        else:
            which = 'smallest'

    mgr.initialize_slepc()

    eps = SLEPc.EPS().create()
    eps.setOperators(H.get_mat())
    eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
    eps.setDimensions(nev)

    eps.setWhichEigenpairs({
        'smallest':SLEPc.EPS.Which.SMALLEST_REAL,
        'exterior':SLEPc.EPS.Which.LARGEST_MAGNITUDE,
        'target':SLEPc.EPS.Which.TARGET_MAGNITUDE,
        }[which])

    if target is None and which=='target':
        raise Exception("Must specify target when setting which='target'")

    if target is not None:
        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        ksp = st.getKSP()
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.CHOLESKY)

    eps.setFromOptions()

    eps.solve()

    nconv = eps.getConverged()

    evals = np.ndarray((nconv,),dtype=np.complex128)

    if getvecs:
        evecs = []

    v = None
    for i in range(nconv):
        if getvecs:
            v = PETSc.Vec().create()
            v.setSizes(1<<H.L)
            v.setFromOptions()
        evals[i] = eps.getEigenpair(i,v)
        if getvecs:
            evecs.append(v)

    if getvecs:
        return (evals,evecs)
    else:
        return evals
