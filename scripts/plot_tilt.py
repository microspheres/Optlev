import numpy as np
import matplotlib.pyplot as plt

tilt = [0, 1000, 2000, 3000, 5000]

xcorr = np.array([-7.24, -6.20, -5.55, -4.60, -7.09])*1e-20

xcorr_err = [1.9e-21, 4.3e-21, 1.2e-20, 1.0e-20, 4.3e-21]

plt.figure()
plt.errorbar(tilt, xcorr, yerr = xcorr_err, fmt = "ro")
plt.show()
