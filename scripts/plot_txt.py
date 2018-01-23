import numpy, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

# plot results from sensitivity_plot2.py

path_sphere = r"C:\data\20171002\bead2_23um_QWP_NS\meas\DC_no_AC_2"

path_noise_laser_off = r"C:\data\20171002\noise_test\laser_off_low_pressure"

path_noise_laser_on = r"C:\data\20171002\noise_test\laser_on_low_pressure"

path_noise_laser_on_lit_open = r"C:\data\20171002\noise_test\laser_on_low_pressure_lits_open"

path_digitalization = r"C:\data\20171002\noise_test\digitalization"

file_sphere = glob.glob(path_sphere+"\*.txt")
path_noise_laser_off = glob.glob(path_noise_laser_off+"\*.txt")
path_noise_laser_on = glob.glob(path_noise_laser_on+"\*.txt")
path_noise_laser_on_lit_open = glob.glob(path_noise_laser_on_lit_open+"\*.txt")
path_digitalization = glob.glob(path_digitalization+"\*.txt")

sphere = np.loadtxt(file_sphere[0])
Nlaser_off = np.loadtxt(path_noise_laser_off[0])
Nlaser_on = np.loadtxt(path_noise_laser_on[0])
Nlaser_on_lit = np.loadtxt(path_noise_laser_on_lit_open[0])
dig = np.loadtxt(path_digitalization[0])

plt.figure()
plt.loglog(sphere[0],sphere[1], label = "23um sphere no AC drive")
plt.loglog(sphere[0],Nlaser_off[1], label = "laser off AC drive")
plt.loglog(sphere[0],Nlaser_on[1], label = "laser on AC drive")
plt.loglog(sphere[0],Nlaser_on_lit[1], label = "laser on lits open AC drive")
plt.loglog(sphere[0], dig[1], label = "digitalization")
plt.legend(loc='upper right')
plt.xlabel("frequency [Hz]")
plt.ylabel("g/$\sqrt{Hz}$")
plt.grid()
plt.show()
