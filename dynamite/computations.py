
from . import initialize
initialize()

import numpy as np
from slepc4py import SLEPc
from petsc4py import PETSc

def evolve(state,H=None,t=None,result=None,tol=None,mfn=None):
    """
    Evolve a quantum state according to the Schrodinger equation
    under the Hamiltonian H. The units are natural, that is, the
    evolution is simply

    .. math::
        \Psi_t = e^{-iHt} \Psi_0

    Parameters
    ----------

    state : petsc4py.PETSc.Vec
        A PETSc vector containing the state to be evolved.
        Can be created easily with :func:`dynamite.tools.build_state`.

    H : Operator
        The Hamiltonian

    t : float
        The time for which to evolve. Can be negative to evolve
        backwards in time.

    result : petsc4py.PETSc.Vec, optional
        Where to store the result state. If not given, a new vector
        is created in which to store the result. If evolving repeatedly
        many times, it is a good idea to pass a result vector to avoid
        repeatedly allocating a lot of memory. Will be overwritten.

    tol : float, optional
        The tolerance for the evolution. Error estimation is difficult
        for Krylov exponentiation; this merely requests that the error
        be somewhat close to ``tol``. There is no guarantee that it will
        actually be smaller.

    mfn : slepc4py.SLEPc.MFN, optional
        Advanced users can pass their own matrix function object from
        SLEPc. In that case the arguments ``H`` and ``t`` can be omitted
        if they have been set for the MFN object manually.

    Returns
    -------
    petsc4py.PETSc.Vec
        The result vector
    """

    if result is None:
        result = H.get_mat().createVecs(side='l')

    if H is not None:
        # check if the evolution is trivial. if H*state = 0, then the evolution does nothing and state is unchanged.
        # In this case MFNSolve fails. to avoid that, we check if we have that condition.
        H.get_mat().mult(state,result)
        if result.norm() == 0:
            result = state.copy()
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

    mfn.solve(state,result)

    return result

def eigsolve(H,getvecs=False,nev=1,which=None,target=None):
    """
    Solve for a subset of the eigenpairs of the Hamiltonian.

    By default, solves for the eigenvalue with the smallest (most
    negative) real part, e.g. the ground state. Which eigenvalues
    are sought and how many can be adjusted with the options.

    .. note::
        Do NOT try to use this function to solve for the whole spectrum!
        It's very efficient at finding a few eigenvalues, but no
        faster than other routines for finding all of them. In the
        future an efficient solver for the whole spectrum may be
        included with dynamite.

    Parameters
    ----------

    getvecs : Bool
        Whether to return eigenvectors as well as eigenvalues.

    nev : int
        The number of eigenvalues sought. The algorithm may
        return more eigenvalues than ``nev`` if more happen to
        converge.

    which : str
        Which eigenvalues to seek. Options are\:

        - ``"smallest"``, to find the eigenvalues with smallest real part

        - ``"exterior"``, to find eigenvalues largest in absolute magnitude

        - ``"target"``, to find eigenvalues closest to the given target

        If ``target`` is set, ``which`` can be omitted and will
        automatically be set to ``"target"``.

    target : float
        Using the shift-invert method, the eigensolver can seek
        the eigenvalues with real part closest to some target value.
        This requires a linear solve and so will be slower than solving
        for exterior eigenvalues.
        PETSc must be configured with a parallel linear solver
        (e.g. ``--download-mumps`` option in ``configure``) to use
        this option in parallel.
    """

    if which is None:
        if target is not None:
            which = 'target'
        else:
            which = 'smallest'

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

        # fix for "bug" discussed here: https://www.mail-archive.com/petsc-users@mcs.anl.gov/msg22867.html
        eps.setOperators(H.get_mat(diag_entries=True))

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
