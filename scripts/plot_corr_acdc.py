import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

C = np.array([-4.99, -4.29, -4.63])*1.0e-20

Cerr = np.array([0.92, 0.76, 2.14])*1.0e-20

ACrDC = np.array([(10/2.)/5., (15/2.)/2.5, (6.7/2)/6.6])

AC = -1.91e-20
ACerr = 1.54e-20

X = np.linspace(0,5,100)

plt.figure()
plt.errorbar(ACrDC, C, yerr = Cerr, fmt = "ro")
plt.axhline(AC, color = "g", linestyle = '-', label = "AC only")
plt.axhline(AC+ACerr, color = "g", linestyle = '--')
plt.axhline(AC-ACerr, color = "g", linestyle = '--')
plt.fill_between(X, AC+ACerr,AC-ACerr,color='g',alpha=.5)
plt.ylabel("corr #e")
plt.xlabel("AC$_{amplitude}$/DC")
plt.xlim(0.15, 3.15)
plt.legend()
plt.grid()
plt.show()
