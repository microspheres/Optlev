import numpy, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

# plot results from sensitivity_plot2.py

path_sphere1 = r"C:\data\acceleration_paper\from_dates\20171002bead2_23um_QWP_NS\acc_psd"
path_sphere2 = r"C:\data\acceleration_paper\from_dates\20171004bead8_23um_QWP_NS\acc_psd"
path_sphere3 = r"C:\data\acceleration_paper\from_dates\20171004bead9_15um_QWP_NS\acc_psd"
path_sphere4 = r"C:\data\acceleration_paper\from_dates\20171011bead9_15um_QWP_NS\acc_psd"
path_sphere5 = r"C:\data\acceleration_paper\from_dates\20171013bead8_5um_QWP_NS\nfft2E15"
path_sphere6 = r"C:\data\acceleration_paper\from_dates\20171013bead11_10um_QWP_NS\acc_psd"

path_noise_laser_off = r"C:\data\20171002\noise_test\laser_off_low_pressure"

# path_noise_laser_on = r"C:\data\20171002\noise_test\laser_on_low_pressure"

# path_noise_laser_on_lit_open = r"C:\data\20171002\noise_test\laser_on_low_pressure_lits_open"

path_digitalization = r"C:\data\20171002\noise_test\digitalization"

file_sphere1 = glob.glob(path_sphere1+"\*.txt")
file_sphere2 = glob.glob(path_sphere2+"\*.txt")
file_sphere3 = glob.glob(path_sphere3+"\*.txt")
file_sphere4 = glob.glob(path_sphere4+"\*.txt")
file_sphere5 = glob.glob(path_sphere5+"\*.txt")
file_sphere6 = glob.glob(path_sphere6+"\*.txt")

path_noise_laser_off = glob.glob(path_noise_laser_off+"\*.txt")
# path_noise_laser_on = glob.glob(path_noise_laser_on+"\*.txt")
# path_noise_laser_on_lit_open = glob.glob(path_noise_laser_on_lit_open+"\*.txt")
path_digitalization = glob.glob(path_digitalization+"\*.txt")

sphere1 = np.loadtxt(file_sphere1[0])
sphere2 = np.loadtxt(file_sphere2[0])
sphere3 = np.loadtxt(file_sphere3[0])
sphere4 = np.loadtxt(file_sphere4[0])
sphere5 = np.loadtxt(file_sphere5[0])
sphere6 = np.loadtxt(file_sphere6[0])

Nlaser_off = np.loadtxt(path_noise_laser_off[0])
# Nlaser_on = np.loadtxt(path_noise_laser_on[0])
# Nlaser_on_lit = np.loadtxt(path_noise_laser_on_lit_open[0])
dig = np.loadtxt(path_digitalization[0])

plt.figure()
plt.loglog(sphere1[0],sphere1[1], label = "10.2ng sphere no AC drive")
plt.loglog(sphere1[0],sphere2[1], label = "11.8ng sphere no AC drive")
plt.loglog(sphere1[0],sphere3[1], label = "2.2ng sphere no AC drive")
plt.loglog(sphere1[0],sphere4[1], label = "2.9ng sphere no AC drive")
plt.loglog(sphere5[0],sphere5[1], label = "0.12ng sphere no AC drive")
plt.loglog(sphere1[0],sphere6[1], label = "1.3ng sphere no AC drive")

plt.loglog(sphere1[0],Nlaser_off[1], label = "laser off AC drive")
# plt.loglog(sphere[0],Nlaser_on[1], label = "laser on AC drive")
# plt.loglog(sphere[0],Nlaser_on_lit[1], label = "laser on lits open AC drive")
plt.loglog(sphere1[0], dig[1], label = "digitalization")
plt.legend(loc='upper right')
plt.xlabel("frequency [Hz]")
plt.ylabel("m/s2/$\sqrt{Hz}$")
plt.grid()
plt.show()
