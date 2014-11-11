"""
Acoustic echo cancellation in white background noise with NLMS.

Consider a scenario where two individuals, John and Emily, are talking over the
Internet. John is using his loudspeakers, which means Emily can hear herself
through John's microphone. The speech signal that Emily hears, is a distorted
version of her own. This is caused by the acoustic path from John's
loudspeakers to his microphone. This path includes attenuated echoes, etc.

Now for the problem!

Emily wishes to cancel the echo she hears from John's microphone. Emily only
knows the speech signal she sends to him, call that u(n), and the speech signal
she receives from him, call that d(n). To successfully remove her own echo
from d(n), she must approximate the acoustic path from John's loudspeakers to
his microphone. This path can be approximated by a FIR filter, which means an
adaptive NLMS FIR filter can be used to identify it. The model which Emily uses
to design this filter looks like this:

      u(n) ------->->------+----------->->-----------
                           |                        |
                  +-----------------+      +------------------+
              +->-| Adaptive filter |      |    John's Room   |
              |   +-----------------+      +------------------+
              |            | -y(n)                  |
              |            |           d(n)         |
      e(n) ---+---<-<------+-----------<-<----------+----<-<---- v(n)

As seen, the signal that is sent to John is also used as input to the adaptive
NLMS filter. The output of the filter, y(n), is subtracted from the signal
received from John, which results in an error signal e(n) = d(n)-y(n). By
feeding the error signal back to the adaptive filter, it can minimize the error
by approximating the impulse response (that is the FIR filter coefficients) of
John's room. Note that so far John's speech signal v(n) has not been taken into
account. If John speaks, the error should equal his speech, that is, e(n)
should equal v(n). For this simple example, however, we assume John is quiet
and v(n) is equal to white Gaussian background noise with zero-mean.

In the following example we keep the impulse response of John's room constant.
This is not required, however, since the advantage of adaptive filters, is that
they can be used to track changes in the impulse response.
"""

import numpy as np
import matplotlib.pyplot as plt
import adaptfilt as adf

# Get u(n) - this is available on github or pypi in the examples folder
u = np.load('speech.npy')

# Generate received signal d(n) using randomly chosen coefficients
coeffs = np.concatenate(([0.8], np.zeros(8), [-0.7], np.zeros(9),
                         [0.5], np.zeros(11), [-0.3], np.zeros(3),
                         [0.1], np.zeros(20), [-0.05]))

d = np.convolve(u, coeffs)

# Add background noise
v = np.random.randn(len(d)) * np.sqrt(5000)
d += v

# Apply adaptive filter
M = 100  # Number of filter taps in adaptive filter
step = 0.1  # Step size
y, e, w = adf.nlms(u, d, M, step, returnCoeffs=True)

# Calculate mean square weight error
mswe = adf.mswe(w, coeffs)

# Plot speech signals
plt.figure()
plt.title("Speech signals")
plt.plot(u, label="Emily's speech signal, u(n)")
plt.plot(d, label="Speech signal from John, d(n)")
plt.grid()
plt.legend()
plt.xlabel('Samples')

# Plot error signal - note how the measurement noise affects the error
plt.figure()
plt.title('Error signal e(n)')
plt.plot(e)
plt.grid()
plt.xlabel('Samples')

# Plot mean squared weight error - note that the measurement noise causes the
# error the increase at some points when Emily isn't speaking
plt.figure()
plt.title('Mean squared weight error')
plt.plot(mswe)
plt.grid()
plt.xlabel('Samples')

# Plot final coefficients versus real coefficients
plt.figure()
plt.title('Real coefficients vs. estimated coefficients')
plt.plot(w[-1], 'g', label='Estimated coefficients')
plt.plot(coeffs, 'b--', label='Real coefficients')
plt.grid()
plt.legend()
plt.xlabel('Samples')

plt.show()
