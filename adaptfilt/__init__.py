"""
Adaptfilt
=========
Adaptive filtering module for Python. For more information please visit
https://github.com/Wramberg/adaptfilt or https://pypi.python.org/pypi/adaptfilt
"""
__version__ = '0.2'
__author__ = "Jesper Wramberg & Mathias Tausen"
__license__ = "MIT"

# Ensure user has numpy
try:
    import numpy
except:
    raise ImportError('Failed to import numpy - please make sure this is\
 available before using adaptfilt')

# Import functions directly into adaptfilt namespace
from lms import lms
from nlms import nlms
from nlmsru import nlmsru
from ap import ap
from misc import mswe


def rundoctests(verbose=False):
    """
    Executes doctests
    """
    import doctest
    import lms as testmod1
    import nlms as testmod2
    import ap as testmod3
    import misc as testmod4
    import nlmsru as testmod5
    lmsres = doctest.testmod(testmod1, verbose=verbose)
    nlmsres = doctest.testmod(testmod2, verbose=verbose)
    apres = doctest.testmod(testmod3, verbose=verbose)
    miscres = doctest.testmod(testmod4, verbose=verbose)
    nlmsrures = doctest.testmod(testmod5, verbose=verbose)
    print '   LMS: ', lmsres
    print '  NLMS: ', nlmsres
    print 'NLMSRU: ', nlmsrures
    print '    AP: ', apres
    print '  MISC: ', miscres
