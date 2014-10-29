"""
Functions used for checking input parameters.
"""


def checkDesiredSignal(d, N, M):
    if len(d) < N+M-1:
        raise ValueError('Desired signal must be >= N+M-1 or len(u)')


def checkNumTaps(M):
    if type(M) is not int:
        raise TypeError('Number of filter taps must be type integer')
    elif M <= 0:
        raise ValueError('Number of filter taps must be greater than 0')


def checkInitCoeffs(c, M):
    if len(c) != M:
        err = 'Length of initial filter coefficients must match filter order'
        raise ValueError(err)


def checkIter(N, maxlen):
    if type(N) is not int:
        raise TypeError('Number of iterations must be type integer')
    elif N > maxlen:
        raise ValueError('Number of iterations must not exceed len(u)-M+1')
    elif N <= 0:
        err = 'Number of iterations must be larger than zero, please increase\
 number of iterations N or length of input u'
        raise ValueError(err)


def checkStep(step):
    if type(step) is not float and type(step) is not int:
        raise TypeError('Step must be type float (or integer)')
    elif step < 0:
        raise ValueError('Step size must non-negative')


def checkLeakage(leak):
    if type(leak) is not float and type(leak) is not int:
        raise TypeError('Leakage must be type float (or integer)')
    elif leak > 1 or leak < 0:
        raise ValueError('0 <= Leakage <= 1 must be satisfied')


def checkRegFactor(eps):
    if type(eps) is not float and type(eps) is not int:
        err = 'Regularization (eps) must be type float (or integer)'
        raise ValueError(err)
    elif eps < 0:
        raise ValueError('Regularization (eps) must non-negative')


def checkProjectOrder(K):
    if type(K) is not int:
        raise TypeError('Projection order must be type integer')
    elif (K <= 0):
        raise ValueError('Projection order must be larger than zero')
