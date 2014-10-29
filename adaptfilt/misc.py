"""
Miscellaneous helper functions.
"""
import numpy as np


def mswe(w, v):
    """
    Calculate mean squared weigth error between estimated and true filter
    coefficients, in respect to iterations.

    Parameters
    ----------
    v : array-like
        True coefficients used to generate desired signal, must be a
        one-dimensional array.
    w : array-like
        Estimated coefficients from adaptive filtering algorithm. Must be an
        N x M matrix where N is the number of iterations, and M is the number
        of filter coefficients.

    Returns
    -------
    mswe : numpy.array
        One-dimensional array containing the mean-squared weigth error for
        every iteration.

    Raises
    ------
    TypeError
        If inputs have wrong dimensions

    Note
    ----
    To use this function with the adaptive filter functions set the optional
    parameter returnCoeffs to True. This will return a coefficient matrix w
    corresponding with the input-parameter w.

    Minimal Working Example
    -----------------------
    >>> import numpy as np
    >>> from nlms import nlms
    >>>
    >>> np.random.seed(1337)
    >>> ulen = 2000
    >>> coeff = np.concatenate(([4], np.zeros(10), [-11], np.zeros(7), [0.7]))
    >>> u = np.random.randn(ulen)
    >>> d = np.convolve(u, coeff)
    >>>
    >>> M = 20  # No. of taps
    >>> step = 1  # Step size
    >>> y, e, w = nlms(u, d, M, step, returnCoeffs=True)
    >>> mswe(w, coeff)
    array([  5.89276161e+00,   5.79121198e+00,   5.13903821e+00, ...,
             4.59606482e-32,   4.59606482e-32,   4.22075548e-32])
    """
    # Ensure inputs are numpy arrays
    w = np.array(w)
    v = np.array(v)
    # Check dimensions
    if(len(w.shape) != 2):
        raise TypeError('Estimated coefficients must be in NxM matrix')
    if(len(v.shape) != 1):
        raise TypeError('Real coefficients must be in 1d array')
    # Ensure equal length between estimated and real coeffs
    N, M = w.shape
    L = v.size
    if(M < L):
        v = v[:-(L-M)]
    elif(M > L):
        v = np.concatenate((v, np.zeros(M-L)))

    # Calculate and return MSWE
    mswe = np.mean((w - v)**2, axis=1)
    return mswe
