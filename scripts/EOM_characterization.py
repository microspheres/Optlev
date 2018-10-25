import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import glob
from scipy.optimize import curve_fit

laser_blocked = 0.0/1e6

Pmax = 0.000366 #see info file

path = r"C:\data\20180123\EOM_characterization"
		
file_list = glob.glob(path+r"\meas3.txt")

a = np.loadtxt(file_list[0], unpack = True)

voltage = a[1]
power = np.array(a[0])/Pmax

power = power - laser_blocked

guess = [0.1, 1.0, 0.000005]

def EOM(x, a, b, off):
    x = x*a
    f = b*(0.5)*(1.-np.cos(x)) + off
    return f

popt, pcov = curve_fit(EOM, voltage[150:250], power[150:250], p0 = guess)

print popt
print pcov

Vfit = np.linspace(-10, 10, 1000)

plt.figure()
plt.plot(voltage, power, "r.")
plt.plot(Vfit, EOM(Vfit, *popt), "k--")

plt.xlabel("Voltage [V]")
plt.ylabel("Transmitted Power")
plt.tight_layout(pad = 0)
plt.grid()
plt.show()
