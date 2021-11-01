"""
Adaptfilt
=========
Adaptive filtering module for Python. For more information please visit
https://github.com/Wramberg/adaptfilt or https://pypi.python.org/pypi/adaptfilt
"""
__version__ = '0.3'
__author__ = "Jesper Wramberg & Mathias Tausen"
__license__ = "MIT"

# Ensure user has numpy
try:
    import numpy
except:
    raise ImportError('Failed to import numpy - please make sure this is\
 available before using adaptfilt')

# Import functions directly into adaptfilt namespace
from .lms import lms
from .nlms import nlms
from .nlmsru import nlmsru
from .ap import ap
from .rls import rls
from .misc import mswe


def __rundoctests__(verbose=False):
    """
    Executes doctests
    """
    import doctest
    lmsres = doctest.testfile("lms.py", verbose=verbose)
    nlmsres = doctest.testfile("nlms.py", verbose=verbose)
    apres = doctest.testfile("ap.py", verbose=verbose)
    miscres = doctest.testfile("misc.py", verbose=verbose)
    nlmsrures = doctest.testfile("nlmsru.py", verbose=verbose)
    print('   LMS: %i passed and %i failed' % \
        (lmsres.attempted-lmsres.failed, lmsres.failed))
    print('  NLMS: %i passed and %i failed' % \
        (nlmsres.attempted-nlmsres.failed, nlmsres.failed))
    print('NLMSRU: %i passed and %i failed' % \
        (nlmsrures.attempted-nlmsrures.failed, nlmsrures.failed))
    print('    AP: %i passed and %i failed' % \
        (apres.attempted-apres.failed, apres.failed))
    print('  MISC: %i passed and %i failed' % \
        (miscres.attempted-miscres.failed, miscres.failed))
