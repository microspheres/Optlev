import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import glob
from scipy.optimize import curve_fit

laser_blocked = 0.43/1e6
laser_blocked2 = 50.5/1e9
laser_blocked3 = 50.5/1e9

path = r"C:\data\20180108"
		
file_list = glob.glob(path+r"\meas_before_chamber.txt")
file_list2 = glob.glob(path+r"\meas_after_chamber2.txt")

file_list3 = glob.glob(path+r"\meas_after_chamber3.txt")

a = np.loadtxt(file_list[0], unpack = True)
a2 = np.loadtxt(file_list2[0], unpack = True)
a3 = np.loadtxt(file_list3[0], unpack = True)

angles = a[1]
power = np.array(a[0])

angles2 = a2[1]
power2 = np.array(a2[0])

angles3 = a3[1]
power3 = np.array(a3[0])

power = power - laser_blocked

power2 = power2 - laser_blocked2

power3 = power3 - laser_blocked3

def qwp(x,A,ph,k, off):
    x = x*(2.0*np.pi)/360.0
    ph = ph*(2.0*np.pi)/360.0
    f = 2.0*A*(np.sin(k*x - ph)*np.cos(k*x - ph))**2.0 + off
    return f

popt, pcov = curve_fit(qwp, angles, power)
popt2, pcov2 = curve_fit(qwp, angles2, power2)
popt3, pcov3 = curve_fit(qwp, angles3, power3)

print popt
print popt2

anglesfit = np.linspace(0, 180, 1800)

Ext = np.min(power)/np.max(power)
print Ext

Ext2 = np.min(power2)/np.max(power2)
print Ext2

plt.figure()
plt.plot(angles, power, "r.")
plt.plot(anglesfit, qwp(anglesfit, *popt), "k-")

plt.plot(angles2, power2, "b.")
plt.plot(anglesfit, qwp(anglesfit, *popt2), "k--")

plt.plot(angles3, power3, "g.")

plt.xlabel("Angle [deg]")
plt.ylabel("Power [arb units]")
plt.tight_layout(pad = 0)
plt.grid()
plt.show()
