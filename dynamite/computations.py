
from . import config
from .states import State

import numpy as np

def evolve(H,state,t,result=None,tol=1E-15,algo='krylov',
           check_trivial=False,ncv=None,mfn=None):
    """
    Evolve a quantum state according to the Schrodinger equation
    under the Hamiltonian H. The units are natural, that is, the
    evolution is simply

    .. math::
        \Psi_t = e^{-iHt} \Psi_0

    Parameters
    ----------

    H : Operator
        The Hamiltonian

    state : dynamite.states.State
        A dynamite State object containing the state to be evolved.

    t : float
        The time for which to evolve. Can be negative to evolve
        backwards in time.

    result : dynamite.states.State, optional
        Where to store the result state. If not given, a new vector
        is created in which to store the result. If evolving repeatedly
        many times, it is a good idea to pass a result vector to avoid
        repeatedly allocating a lot of memory. Will be overwritten.

    tol : float, optional
        The tolerance for the evolution. Error estimation is difficult
        for Krylov exponentiation; this merely requests that the error
        be somewhat close to ``tol``. There is no guarantee that it will
        actually be smaller.

    algo : string, optional
        Allowed options: 'krylov' or 'expokit'. Which SLEPc algorithm to
        use to compute the matrix exponential.

    check_trivial : bool, optional
        A switch whether to check whether the evolution is trivial. If
        H*s == 0, evolution fails, but this just means that the evolution
        is the identity. A check for this takes a (short) time.

    ncv : int, optional
        The Krylov subspace size to use. Increasing subspace size can
        increase performance by reducing the number of iterations necessary,
        but also linearly increases memory usage and the number of matrix
        multiplies performed. Optimizing this parameter can significantly
        affect performance.

    mfn : slepc4py.SLEPc.MFN, optional
        Advanced users can pass their own matrix function object from
        SLEPc. In that case the arguments ``H`` and ``t`` can be omitted
        if they have been set for the MFN object manually.

    Returns
    -------
    dynamite.states.State
        The result state
    """

    config.initialize()
    from slepc4py import SLEPc

    if H.L != state.L:
        raise ValueError('Hamiltonian and state have incompatible lengths (H:%d, state:%d)'%
                         (H.L,state.L))

    if result is None:
        result = State(L=H.L,subspace=H.left_subspace)

    if check_trivial:
        # check if the evolution is trivial. if H*state = 0,
        # then the evolution does nothing and state is unchanged.
        # In this case MFNSolve fails. to avoid that, we check if
        # we have that condition.

        H.get_mat().mult(state.vec,result.vec)
        if result.vec.norm() == 0:
            state.vec.copy(result.vec)
            return result

    if mfn is None:

        mfn = SLEPc.MFN().create()
        mfn.setType(algo)

        f = mfn.getFN()
        f.setType(SLEPc.FN.Type.EXP)

        if ncv is not None:
            mfn.setDimensions(ncv)

        mfn.setFromOptions()

    if tol is not None:
        mfn.setTolerances(tol)

    mfn.setOperator(H.get_mat())

    f = mfn.getFN()
    f.setScale(-1j*t)

    mfn.solve(state.vec,result.vec)

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

        - ``"smallest"``, to find the eigenvalues with smallest real part (i.e. most negative)

        - ``"largest"``, to find the eigenvalues with largest real part (i.e. most positive)

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

    Returns
    -------
    numpy.array or tuple(numpy.array, list(dynamite.states.State))
        Either a 1D numpy array of eigenvalues, or a pair containing that array
        and a list of the corresponding eigenvectors.
    """

    config.initialize()
    from slepc4py import SLEPc

    if which is None:
        if target is not None:
            which = 'target'
        else:
            which = 'smallest'

    eps = SLEPc.EPS().create()
    eps.setProblemType(SLEPc.EPS.ProblemType.HEP)

    if target is not None:
        # shift-invert not supported for shell matrices
        # TODO: can use a different preconditioner so shift-invert works!
        if H.shell:
            raise TypeError('Shift-invert ("target") not supported for shell matrices.')

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)

        eps.setTarget(target)

        # fix for "bug" discussed here:
        # https://www.mail-archive.com/petsc-users@mcs.anl.gov/msg22867.html
        eps.setOperators(H.get_mat(diag_entries=True))
    else:
        eps.setOperators(H.get_mat())

    eps.setDimensions(nev)

    eps.setWhichEigenpairs({
        'smallest':SLEPc.EPS.Which.SMALLEST_REAL,
        'largest':SLEPc.EPS.Which.LARGEST_REAL,
        'exterior':SLEPc.EPS.Which.LARGEST_MAGNITUDE,
        'target':SLEPc.EPS.Which.TARGET_MAGNITUDE,
        }[which])

    if target is None and which=='target':
        raise ValueError("Must specify target when setting which='target'")

    eps.setFromOptions()

    eps.solve()

    nconv = eps.getConverged()

    evals = np.ndarray((nconv,),dtype=np.complex128)

    if getvecs:
        evecs = []

    v = None
    for i in range(nconv):
        if getvecs:
            v = State(H.L,H.subspace)
        evals[i] = eps.getEigenpair(i,v.vec)
        if getvecs:
            evecs.append(v)

    if getvecs:
        return (evals,evecs)
    else:
        return evals

def reduced_density_matrix(state,cut_size,fillall=True):
    """
    Compute the reduced density matrix of a state vector by
    tracing out some set of spins. Currently only supports
    tracing out a certain number of spins, but not specifying
    which will be removed. (Spins with indices 0 to cut_size-1
    are traced out in the current implementation).

    The density matrix is returned on process 0, the function
    returns ``None`` on all other processes.

    Parameters
    ----------

    state : dynamite.states.State
        A dynamite State object.

    cut_size : int
        The number of spins to keep. So, for ``cut_size``:math:`=n`,
        :math:`L-n` spins will be traced out, resulting in a density
        matrix of dimension :math:`2^n {\\times} 2^n`.

    fillall : bool,optional
        Whether to fill the whole matrix. Since it will be Hermitian,
        only the lower triangle is necessary to describe the whole matrix.
        If this option is set to true, only the lower triangle will be filled.

    Returns
    -------
    numpy.ndarray[np.complex128]
        The density matrix
    """

    config.initialize()
    from ._backend.bpetsc import reduced_density_matrix as backend_rdm

    if cut_size != int(cut_size) or not 0 <= cut_size <= state.L:
        raise ValueError('cut_size must be an integer between 0 and L, inclusive.')

    return backend_rdm(state.vec,cut_size,fillall=fillall)

def entanglement_entropy(state,cut_size):
    """
    Compute the entanglement of a state across some cut on the
    spin chain. To be precise, this is the bipartite entropy of
    entanglement.

    Currently, this quantity is computed entirely on process 0.
    As a result, the function returns ``-1`` on all other processes.
    Ideally, at some point a parallelized dense matrix solver will
    be used in this computation.

    Parameters
    ----------

    state : dynamite.states.State
        A dynamite State object.

    cut_size : int
        The number of spins on one side of the cut. To be precise,
        the cut will be made between the spins at index ``cut_size-1``
        and ``cut_size``.

    Returns
    -------

    float
        The entanglement entropy
    """

    reduced = reduced_density_matrix(state,cut_size,fillall=False)

    # currently everything computed on process 0
    if reduced is None:
        return -1

    w = np.linalg.eigvalsh(reduced)
    EE = -np.sum(w * np.log(w,where=w>0))

    return EE

def get_tstep(ncv,nrm,tol=1E-7):
    """
    Compute the length of a sub-step in a Expokit matrix
    exponential solve.
    """
    f = ((ncv+1)/2.72)**(ncv+1) * np.sqrt(2*np.pi*(ncv+1))
    t = ((1/nrm)*(f*tol)/(4.0*nrm))**(1/ncv)
    s = 10.0**(np.floor(np.log10(t))-1)
    return np.ceil(t/s)*s

def estimate_compute_time(t,ncv,nrm,tol=1E-7):
    """
    Estimate compute time in units of matrix multiplies, for
    an expokit exponential solve.
    """
    tstep = get_tstep(ncv,nrm,tol)
    iters = np.ceil(t/tstep)
    return ncv*iters
