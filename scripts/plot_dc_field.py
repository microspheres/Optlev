import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

F = np.array([2.59, 4.52, 7.25, 10.33, 14.05, 18.48, 23.4, 30.1, 37.0, 57.9, 76.7])

DC = np.array([298.5, 398.2, 497.9, 597.6, 697.3, 796.9, 896.6, 996.3, 1096.0, 1394.8, 1594.0])

def func(x,a,b,c):
    return a*x*x + 0*b*x + 0*c

popt, pcov = opt.curve_fit(func, DC, F)

print popt
print func(279,*popt)

plt.figure()
plt.plot(DC,F, "o")
plt.plot(DC, func(DC,*popt))
plt.show()
