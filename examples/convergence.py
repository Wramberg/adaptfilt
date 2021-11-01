"""
Convergence comparison of different adaptive filtering algorithms (with
different step sizes) in white Gaussian noise.
"""

import numpy as np
import matplotlib.pyplot as plt
import adaptfilt as adf

# Generating inpud and desired signal
N = 3000
coeffs = np.concatenate(([-4, 3.2], np.zeros(20), [0.7], np.zeros(33), [-0.1]))
u = np.random.randn(N)
d = np.convolve(u, coeffs)

# Perform filtering
M = 60  # No. of taps to estimate
mu1 = 0.0008  # Step size 1 in LMS
mu2 = 0.0004  # Step size 1 in LMS
beta1 = 0.08  # Step size 2 in NLMS and AP
beta2 = 0.04  # Step size 2 in NLMS and AP
ffactor1 = 0.995  # Forgetting factor 1 in RLS
ffactor2 = 0.8  # Forgetting factor 2 in RLS
K = 3  # Projection order 1 in AP

# LMS
y_lms1, e_lms1, w_lms1 = adf.lms(u, d, M, mu1, returnCoeffs=True)
y_lms2, e_lms2, w_lms2 = adf.lms(u, d, M, mu2, returnCoeffs=True)
mswe_lms1 = adf.mswe(w_lms1, coeffs)
mswe_lms2 = adf.mswe(w_lms2, coeffs)

# NLMS
y_nlms1, e_nlms1, w_nlms1 = adf.nlms(u, d, M, beta1, returnCoeffs=True)
y_nlms2, e_nlms2, w_nlms2 = adf.nlms(u, d, M, beta2, returnCoeffs=True)
mswe_nlms1 = adf.mswe(w_nlms1, coeffs)
mswe_nlms2 = adf.mswe(w_nlms2, coeffs)

# AP
y_ap1, e_ap1, w_ap1 = adf.ap(u, d, M, beta1, K, returnCoeffs=True)
y_ap2, e_ap2, w_ap2 = adf.ap(u, d, M, beta2, K, returnCoeffs=True)
mswe_ap1 = adf.mswe(w_ap1, coeffs)
mswe_ap2 = adf.mswe(w_ap2, coeffs)

# RLS
y_rls1, e_rls1, w_rls1 = adf.rls(u, d, M, ffactor1, returnCoeffs=True)
y_rls2, e_rls2, w_rls2 = adf.rls(u, d, M, ffactor2, returnCoeffs=True)
mswe_rls1 = adf.mswe(w_rls1, coeffs)
mswe_rls2 = adf.mswe(w_rls2, coeffs)

# Plot results
plt.figure()
plt.title('Convergence comparison of different adaptive filtering algorithms')
plt.plot(mswe_lms1, 'b', label='LMS with stepsize=%.4f' % mu1)
plt.plot(mswe_lms2, 'b--', label='LMS with stepsize=%.4f' % mu2)
plt.plot(mswe_nlms1, 'g', label='NLMS with stepsize=%.2f' % beta1)
plt.plot(mswe_nlms2, 'g--', label='NLMS with stepsize=%.2f' % beta2)
plt.plot(mswe_ap1, 'r', label='AP with stepsize=%.2f' % beta1)
plt.plot(mswe_ap2, 'r--', label='AP with stepsize=%.2f' % beta2)
plt.plot(mswe_rls1, 'k', label='RLS with forgetting factor=%.3f' % ffactor1)
plt.plot(mswe_rls2, 'k--', label='RLS with forgetting factor=%.3f' % ffactor2)
plt.legend()
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Mean-squared weight error')
plt.show()
