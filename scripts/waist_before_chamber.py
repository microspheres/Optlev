import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

position = np.arange(0,28,1)

position = 5.*(0.0254)*np.array(position) # in mm

power = (1.26, 1.26, 1.25, 1.24, 1.23, 1.21, 1.18, 1.14, 1.09, 1.04, 0.96, 0.86, 0.74, 0.63, 0.51, 0.41, 0.32, 0.24, 0.18, 0.14, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02) # in mW

def gauss(x,A,m,s):
    g = A*np.exp(-2.*((x-m)/s)**2)
    return g

print len(power)
print len(position)

popt, pcov = curve_fit(gauss, position, np.abs(np.gradient(power)))

dd = np.linspace(position[0], position[-1],1000)

print popt
print np.sqrt(pcov[2][2])

plt.figure()
plt.plot(position, np.abs(np.gradient(power)), "ro")
plt.plot(dd, gauss(dd,*popt))
# plt.gca().set_yscale("log")
plt.show()
