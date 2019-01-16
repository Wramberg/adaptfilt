import numpy as np
import adaptfilt._paramcheck as _pchk


def rls(u, d, M, ffactor, initP=None, initCoeffs=None, N=None,
        returnCoeffs=False):
    """
    Perform recursive least-squares adaptive filtering with exponential weight
    function. The signal u is filtered to minimize the error given by e=d-y,
    where y is the output of the adaptive filter.

    Parameters
    ----------
    u : array-like
        One-dimensional filter input.
    d : array-like
        One-dimensional desired signal, i.e., the output of the unknown FIR
        system which the adaptive filter should identify. Must have length >=
        len(u), or N+M-1 if number of iterations are limited (via the N
        parameter).
    M : int
        Desired number of filter taps (desired filter order + 1), must be
        non-negative.
    ffactor : float
        Forgetting factor of the exponential weight function. Must lie in the
        interval ]0:1] for the RLS algorithm to be stable in the mean and
        mean-square.

    Optional Parameters
    -------------------
    initCoeffs : array-like
        Initial filter coefficients to use. Should match desired number of
        filter taps, defaults to zeros.
    N : int
        Number of iterations to run. Must be less than or equal to len(u)-M+1.
        Defaults to len(u)-M+1.
    initP : array-like
        M x M matrix of initial values for the inverse correlation matrix. If
        None, a scaled identity matrix is used - this introduces bias. TODO:
        estimate correlation matrix instead
    returnCoeffs : boolean
        If true, will return all filter coefficients for every iteration in an
        N x M matrix. Does not include the initial coefficients. If false, only
        the latest coefficients in a vector of length M is returned. Defaults
        to false.

    Returns
    -------
    y : numpy.array
        Output values of LMS filter, array of length N.
    e : numpy.array
        Error signal, i.e, d-y. Array of length N.
    w : numpy.array
        Final filter coefficients in array of length M if returnCoeffs is
        False. NxM array containing all filter coefficients for all iterations
        otherwise.

    Raises
    ------
    TypeError
        If number of filter taps M is not type integer, number of iterations N
        is not type integer, or forgetting factor is not type float/int.
    ValueError
        If number of iterations N is greater than len(u)-M, number of filter
        taps M is negative, or if ffactor is outside allowed range.
    """

    # Num taps check
    _pchk.checkNumTaps(M)
    # Max iteration check
    if N is None:
        N = len(u)-M+1
    _pchk.checkIter(N, len(u)-M+1)
    # Check len(d)
    _pchk.checkDesiredSignal(d, N, M)
    # Forgetting factor check
    _pchk.checkForgettingFactor(ffactor)
    # Initial inverse correlation matrix check
    if initP is None:
        # TODO: estimate correlation matrix and cross correlation vector and use
        # for initialization of P and w
        initP = np.identity(M) * 0.5
    else:
        _pchk.checkInverseCorrelationMatrix(initP, M)
    # Init. coeffs check
    if initCoeffs is None:
        initCoeffs = np.zeros((M, 1))
    else:
        _pchk.checkInitCoeffs(initCoeffs, M)

    # Ensure that u is a column vector
    u = u.reshape((u.shape[0], 1))

    # Initialization
    w = initCoeffs.reshape((M, 1))  # ensure column vector
    y = np.empty(N)
    xi = np.empty(N)

    P = initP

    if returnCoeffs:
        W = np.zeros((N, M))  # Matrix to hold coeffs for each iteration

    # Perform filtering
    for n in range(N):
        # Slice M latest data points
        x = np.flipud(u[n:n+M])

        # Filter using current coeffs, calculate a priori error
        y[n] = np.dot(x.T, w)
        xi[n] = d[n+M-1] - y[n]

        # Solve the weighted normal equations using recursive approach
        pi = np.dot(P, x)
        # Note: For low values of ffactor, the denominator of this gets very
        # close to zero. TODO: implement some sort of stabilization
        k = pi / (ffactor + np.dot(x.T, pi))
        w = w + k * xi[n]

        # Update P matrix for next iteration
        P = 1./ffactor * (P - np.dot(k, pi.T))

        if returnCoeffs:
            W[n, :] = w[:, 0]

    if returnCoeffs:
        w = W

    return y, xi, w
