import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab



X1 = [0., 20., 40., 60., 80., 100.]

g1 = -np.array([2.32, 2.45, 1.61, 0.57, 0.75, 1.80])# actually this isn't g this is corr(2 omega)

dg1 = np.array([0.06, 0.07, 0.04, 0.04, 0.09, 0.06]) # error bar

# norm2 = np.amin(g2)

# g1 = g2/norm2
# dg1 = dg2/norm2

def para(x,a,b,c):
    return a*(x-b)**2 + c

popt1, pcov1 = curve_fit(para, X1, g1, sigma = dg1)

plt.figure()

A = np.arange(110)

plt.errorbar(X1, g1, yerr = dg1, fmt='o')
plt.plot(A, para(A, *popt1))
plt.xlabel('piezo_X')
plt.ylabel('$\propto$ dE/E')
plt.grid()

plt.show()
