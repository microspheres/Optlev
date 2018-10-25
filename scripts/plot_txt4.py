import numpy, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
from scipy.optimize import curve_fit

# plot results from sensitivity_plot2.py

path_sphere1 = r"C:\data\acceleration_paper\from_dates\20171004bead9_15um_QWP_NS\1mbar\XYcool\acc"
path_sphere2 = r"C:\data\acceleration_paper\from_dates\20171004bead9_15um_QWP_NS\1mbar\XYZcool\acc"
path_sphere3 = r"C:\data\acceleration_paper\from_dates\20171004bead9_15um_QWP_NS\nfft2E19\acc"
path_sphere4 = r"C:\data\acceleration_paper\from_dates\20171004bead9_15um_QWP_NS\pointnoise\acc_in_ug"

path_noise_laser_off = r"C:\data\20171002\noise_test\laser_off_low_pressure"

# path_noise_laser_on = r"C:\data\20171002\noise_test\laser_on_low_pressure"

# path_noise_laser_on_lit_open = r"C:\data\20171002\noise_test\laser_on_low_pressure_lits_open"

path_digitalization = r"C:\data\20171002\noise_test\digitalization"

file_sphere1 = glob.glob(path_sphere1+"\*.txt")
file_sphere2 = glob.glob(path_sphere2+"\*.txt")
file_sphere3 = glob.glob(path_sphere3+"\*.txt")
file_sphere4 = glob.glob(path_sphere4+"\*.txt")

path_noise_laser_off = glob.glob(path_noise_laser_off+"\*.txt")
# path_noise_laser_on = glob.glob(path_noise_laser_on+"\*.txt")
# path_noise_laser_on_lit_open = glob.glob(path_noise_laser_on_lit_open+"\*.txt")
path_digitalization = glob.glob(path_digitalization+"\*.txt")

sphere1 = np.loadtxt(file_sphere1[0])
sphere2 = np.loadtxt(file_sphere2[0])
sphere3 = np.loadtxt(file_sphere3[0])
sphere4 = np.loadtxt(file_sphere4[0])

Nlaser_off = np.loadtxt(path_noise_laser_off[0])
# Nlaser_on = np.loadtxt(path_noise_laser_on[0])
# Nlaser_on_lit = np.loadtxt(path_noise_laser_on_lit_open[0])
dig = np.loadtxt(path_digitalization[0])

def func(x,x0,A,d):
    f = A/(x*np.sqrt(d**2 + ((x**2-x0**2)/x)**2))
    return f

fit = np.logical_and(sphere2[0]>40, sphere2[0]<1000)
popt, pcov = curve_fit(func, sphere2[0][fit], sphere2[1][fit], p0 = [220,1e4,0.1])
# popt = [220,1e4,0.1]

plt.figure()
# plt.loglog(sphere2[0], 20*1e6*func(sphere2[0],*popt)/9.8)

plt.loglog(sphere1[0],20*(sphere1[1]/9.8)*10**6, label = "1mbar no feedback", color = "k")
plt.loglog(sphere2[0],20*(sphere2[1]/9.8)*10**6, label = "1mbar with feedback", color = [0.2,0.2,1])
plt.loglog(sphere3[0],(sphere3[1]/9.8)*10**6, label = "<$10^{-6}$mbar with feedback", color = "r", lw = 1)
# plt.loglog(sphere4[0],sphere4[1], lw = 1, color = "gray" ,alpha = 0.6, label = "Pointing Noise with feedback")
plt.loglog(sphere4[0],20*1e6*sphere4[1]*func(sphere4[0],*popt)/(9.8*60.), lw = 1, color = "gray" ,alpha = 0.6, label = "Pointing Noise with feedback")

# plt.loglog(Nlaser_off[0],Nlaser_off[1], label = "laser off AC drive")
# plt.loglog(sphere[0],Nlaser_on[1], label = "laser on AC drive")
# plt.loglog(sphere[0],Nlaser_on_lit[1], label = "laser on lits open AC drive")
# plt.loglog(dig[0], dig[1], label = "digitalization")
plt.legend(loc='lower left', frameon = False)
plt.xlabel("Frequency [Hz]", fontsize = 17)
plt.ylabel("Acceleration Sensitivity [$\mu$g/$\sqrt{Hz}$]", fontsize = 17)
plt.xlim(1, 1000)
# plt.ylim(0.1, 1000)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.grid()
plt.tight_layout(pad = 0)
plt.show()
