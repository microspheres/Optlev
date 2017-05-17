import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab

X = [0., 30., 40., 56.9, 90]

# mW
g = [0.39, 0.35, 0.020, 0.19, 0.79]

dg = [0.029, 0.029, 0.032, 0.036, 0.046]


def para(x, A, B, C):
    return A*(x-B)**2 + C

popt, pcov = curve_fit(para, X, g, sigma = dg)

A = 1.*np.arange(100)

plt.figure(1)
#plt.plot(X, g,'ro', color="red")
plt.errorbar(X, g, yerr = dg, fmt='o' , color="red")
plt.plot(A, para(A, *popt))
plt.xlabel('piezo_X')
plt.ylabel('$\propto$ dE/E')
plt.grid()

plt.show()
