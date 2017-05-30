import numpy as np
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab
import os
import matplotlib.pyplot as plt

path = r'C:\data\20170511\Adam_simulation'

file_name = 'gradient_angle.txt'

a = np.loadtxt(os.path.join(path, file_name),unpack=True)

def parabola(x, a, b, c):
    return a*(x-b)**2 + c

popt, pcov = curve_fit(parabola, a[0], a[1], bounds=([-2000., 7.e-6, 5.e-4], [-100., 1.e-5, 1.e-3]))

b = np.diff(a[1])/np.diff(a[0])

plt.figure()
plt.plot(a[0],a[1],'.')
plt.plot(a[0], parabola(a[0], *popt))
plt.xlabel("$\\theta$(rad)")
plt.ylabel("dE/E")
plt.show()

print popt
print 
