import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ang = np.arange(0.,100.,5.)

ang2 = np.arange(0.,105.,5.)

P1 = [21.0, 48.1, 96.5, 161.0, 234.2, 305.8, 366.9, 411.3, 433.5, 430.2, 402.5, 352.6, 288.6, 218.4, 147.6, 86.7, 43.0, 20.4, 21.7, 47.0]

P1 = np.array(P1)


P2 = [44.3, 98.3, 188.9, 299.6, 407.8, 485.5, 510.8, 482.9, 408.4, 304.5, 201.8, 116.6, 58.8, 26.7, 10.8, 4.6, 2.4, 1.7, 1.8, 3.2]

P2 = np.array(P2)

P3 = [39.6, 96.2, 171.7, 260.1, 353.6, 440.8, 511.5, 561.7, 584.3, 576.5, 540.0, 477.1, 396.5, 307.2, 215.1, 131.9, 66.0, 21.5, 1.8, 7.5, 35.0]

P3 = np.array(P3)


P4 = [27.3, 53.2, 84.1, 117.5, 150.7, 179.3, 200.5, 210.3, 210.0, 199.4, 179.9, 150.7, 117.1, 82.9, 51.6, 24.8, 8.5, 1.1, 3.7, 15.8, 34.3]

P4 = np.array(P4)


def qwp(x,A,ph,k):
    x = x*(2.0*np.pi)/360.0
    ph = ph*(2.0*np.pi)/360.0
    f = 2.0*A*(np.sin(k*x - ph)*np.cos(k*x-ph))**2.0
    return f

a01 = (400.0, 0.0, 1.0)
popt, pcov = curve_fit(qwp, ang, P1, a01)

a02 = (1200.0, -5.0, 0.78)
popt2, pcov2 = curve_fit(qwp, ang2, P3, a02)

a04 = (200.0, -5.0, 0.78)
popt4, pcov4 = curve_fit(qwp, ang2, P4, a02)

angles =  np.linspace(0, 95, 3000)

angles2 =  np.linspace(0, 100, 3000)

print popt4
print pcov4

plt.figure()
plt.plot(ang, P1, 'ro')
plt.plot(angles, qwp(angles,*popt), "k-")

plt.plot(ang2, P3, 'bo')
plt.plot(angles2, qwp(angles2,*popt2), "g-")

plt.plot(ang2, P4, 'rx')
plt.plot(angles2, qwp(angles2,*popt4), "b-")

plt.xlabel("angle[deg]")
plt.ylabel("Power[uW]")
plt.grid()
plt.show()
