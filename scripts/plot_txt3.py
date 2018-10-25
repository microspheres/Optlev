import numpy, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

# plot results from sensitivity_plot2.py

path_sphere1 = r"C:\data\acceleration_paper\from_dates\20171011bead9_15um_QWP_NS\1mbar\XYcool\acc"
path_sphere2 = r"C:\data\acceleration_paper\from_dates\20171011bead9_15um_QWP_NS\1mbar\XYZcool\acc"
path_sphere3 = r"C:\data\acceleration_paper\from_dates\20171011bead9_15um_QWP_NS\nfft2E19\acc"

path_noise_laser_off = r"C:\data\20171002\noise_test\laser_off_low_pressure"

# path_noise_laser_on = r"C:\data\20171002\noise_test\laser_on_low_pressure"

# path_noise_laser_on_lit_open = r"C:\data\20171002\noise_test\laser_on_low_pressure_lits_open"

path_digitalization = r"C:\data\20171002\noise_test\digitalization"

file_sphere1 = glob.glob(path_sphere1+"\*.txt")
file_sphere2 = glob.glob(path_sphere2+"\*.txt")
file_sphere3 = glob.glob(path_sphere3+"\*.txt")

path_noise_laser_off = glob.glob(path_noise_laser_off+"\*.txt")
# path_noise_laser_on = glob.glob(path_noise_laser_on+"\*.txt")
# path_noise_laser_on_lit_open = glob.glob(path_noise_laser_on_lit_open+"\*.txt")
path_digitalization = glob.glob(path_digitalization+"\*.txt")

sphere1 = np.loadtxt(file_sphere1[0])
sphere2 = np.loadtxt(file_sphere2[0])
sphere3 = np.loadtxt(file_sphere3[0])

Nlaser_off = np.loadtxt(path_noise_laser_off[0])
# Nlaser_on = np.loadtxt(path_noise_laser_on[0])
# Nlaser_on_lit = np.loadtxt(path_noise_laser_on_lit_open[0])
dig = np.loadtxt(path_digitalization[0])

plt.figure()
plt.loglog(sphere1[0],(sphere1[1]/9.8)*10**6, label = "1mbar no feedback", color = "k")
plt.loglog(sphere2[0],(sphere2[1]/9.8)*10**6, label = "1mbar with feedback", color = [0.2,0.2,1])
plt.loglog(sphere3[0],(sphere3[1]/9.8)*10**6, label = "<$10^{-6}$mbar with feedback", color = "r", lw = 1)

# plt.loglog(Nlaser_off[0],Nlaser_off[1], label = "laser off AC drive")
# plt.loglog(sphere[0],Nlaser_on[1], label = "laser on AC drive")
# plt.loglog(sphere[0],Nlaser_on_lit[1], label = "laser on lits open AC drive")
# plt.loglog(dig[0], dig[1], label = "digitalization")
plt.legend(loc='lower left')
plt.xlabel("Frequency [Hz]", fontsize = 17)
plt.ylabel("Acceleration Sensitivity [$\mu$g/$\sqrt{Hz}$]", fontsize = 17)
plt.xlim(10, 1000)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.grid()
plt.tight_layout(pad = 0)
plt.show()
