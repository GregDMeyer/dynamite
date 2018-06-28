
from . import config
from .states import State

import numpy as np

def evolve(H, state, t, result=None, **kwargs):
    r"""
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

    ncv : int, optional
        The Krylov subspace size to use. Increasing subspace size can
        increase performance by reducing the number of iterations necessary,
        but also linearly increases memory usage and the number of matrix
        multiplies performed. Optimizing this parameter can significantly
        affect performance.

    Returns
    -------
    dynamite.states.State
        The result state
    """

    config.initialize()
    from slepc4py import SLEPc

    if H.right_subspace != state.subspace:
        raise ValueError('Hamiltonian and state are defined on different '
                         'subspaces.')

    if result is None:
        result = State(L=H.L, subspace=H.left_subspace)
    elif H.left_subspace != result.subspace:
        raise ValueError('Hamiltonian and result state are defined on different '
                         'subspaces.')

    mfn = SLEPc.MFN().create()
    f = mfn.getFN()
    f.setType(SLEPc.FN.Type.EXP)
    f.setScale(-1j*t)

    if 'algo' in kwargs:
        mfn.setType(kwargs['algo'])

    if 'ncv' in kwargs:
        mfn.setDimensions(kwargs['ncv'])

    if 'tol' in kwargs:
        mfn.setTolerances(kwargs['tol'])

    mfn.setFromOptions()
    mfn.setOperator(H.get_mat())

    mfn.solve(state.vec,result.vec)

    return result

def eigsolve(H, getvecs=False, nev=1, which='smallest', target=None, tol=None):
    r"""
    Solve for a subset of the eigenpairs of the Hamiltonian.

    By default, solves for the eigenvalue with the smallest (most
    negative) real part, e.g. the ground state. Which eigenvalues
    are sought and how many can be adjusted with the options.

    .. note::
        Krylov algorithms have difficulty with degenerate or very nearly degenerate
        eigenvalues. Degenerate eigenvalues may be missed, and near-degenerate
        eigenstates may be inaccurate.

    .. note::
        Do not try to use this function to solve for the whole spectrum!
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

    tol : float
        The tolerance for the computation.

    Returns
    -------
    numpy.array or tuple(numpy.array, list(dynamite.states.State))
        Either a 1D numpy array of eigenvalues, or a pair containing that array
        and a list of the corresponding eigenvectors.
    """

    config.initialize()
    from slepc4py import SLEPc

    eps = SLEPc.EPS().create()
    eps.setProblemType(SLEPc.EPS.ProblemType.HEP)

    if target is not None:
        # shift-invert not supported for shell matrices
        # TODO: can use a different preconditioner so shift-invert works!
        which = 'target'

        if H.shell:
            raise TypeError('Shift-invert ("target") not supported for shell matrices.')

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        eps.setTarget(target)

        # fix for "bug" discussed here:
        # https://www.mail-archive.com/petsc-users@mcs.anl.gov/msg22867.html
        eps.setOperators(H.get_mat(diag_entries=True))
    else:
        if which=='target':
            raise ValueError("Must specify target when setting which='target'")
        eps.setOperators(H.get_mat())

    eps.setDimensions(nev)

    eps.setWhichEigenpairs({
        'smallest':SLEPc.EPS.Which.SMALLEST_REAL,
        'largest':SLEPc.EPS.Which.LARGEST_REAL,
        'exterior':SLEPc.EPS.Which.LARGEST_MAGNITUDE,
        'target':SLEPc.EPS.Which.TARGET_MAGNITUDE,
        }[which])

    eps.setTolerances(tol = tol)

    eps.setFromOptions()
    eps.solve()
    nconv = eps.getConverged()

    evals = np.ndarray((nconv,), dtype=np.float)
    evecs = []

    for i in range(nconv):
        evals[i] = eps.getEigenpair(i, None).real
        if getvecs:
            v = State(H.L, H.subspace)
            eps.getEigenpair(i, v.vec)
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
    from ._backend import bpetsc

    if cut_size != int(cut_size) or not 0 <= cut_size <= state.L:
        raise ValueError('cut_size must be an integer between 0 and L, inclusive.')

    return bpetsc.reduced_density_matrix(state.vec, cut_size, fillall=fillall)

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

    reduced = reduced_density_matrix(state, cut_size, fillall=False)

    # currently everything computed on process 0
    if reduced is None:
        return -1

    w = np.linalg.eigvalsh(reduced)
    EE = -np.sum(w * np.log(w, where=w>0))

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
