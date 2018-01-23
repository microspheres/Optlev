import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

position = (0, 24, 30, 35, 40, 45, 50)

position = (0.0254)*np.array(position) # in mm

power = (43.5, 42.8, 41.3, 32.3, 7.3, 1.6, 0.38) # in mW

def gauss(x,A,m,s):
    g = A*np.exp(-((x-m)/s)**2)
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
