import numpy, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
from scipy.optimize import curve_fit
import matplotlib.patches as patches

# plot results from sensitivity_plot2.py

path_sphere1 = r"C:\data\acceleration_paper\from_dates\20171031bead1_15um_QWP_NS\1mbar\YZ"
path_sphere2 = r"C:\data\acceleration_paper\from_dates\20171031bead1_15um_QWP_NS\1mbar\XYZ"
path_sphere3 = r"C:\data\acceleration_paper\from_dates\20171031bead1_15um_QWP_NS\accnfft19"
path_sphere4 = r"C:\data\acceleration_paper\from_dates\20171031bead1_15um_QWP_NS\pointing_noise_low_pressure\psd"
path_sphere5 = r"C:\data\acceleration_paper\from_dates\20171031bead1_15um_QWP_NS\whitenoise\psd"


path_noise_laser_off = r"C:\data\20171002\noise_test\laser_off_low_pressure"

path_noise_laser_on = r"C:\data\20171002\noise_test\laser_on_low_pressure"

# path_noise_laser_on_lit_open = r"C:\data\20171002\noise_test\laser_on_low_pressure_lits_open"

path_digitalization = r"C:\data\20171002\noise_test\digitalization"

file_sphere1 = glob.glob(path_sphere1+"\*.txt")
file_sphere2 = glob.glob(path_sphere2+"\*.txt")
file_sphere3 = glob.glob(path_sphere3+"\*.txt")
file_sphere4 = glob.glob(path_sphere4+"\*.txt")
file_sphere5 = glob.glob(path_sphere5+"\*.txt")

path_noise_laser_off = glob.glob(path_noise_laser_off+"\*.txt")
path_noise_laser_on = glob.glob(path_noise_laser_on+"\*.txt")
# path_noise_laser_on_lit_open = glob.glob(path_noise_laser_on_lit_open+"\*.txt")
path_digitalization = glob.glob(path_digitalization+"\*.txt")

sphere1 = np.loadtxt(file_sphere1[0])
sphere2 = np.loadtxt(file_sphere2[0])
sphere3 = np.loadtxt(file_sphere3[0])
sphere4 = np.loadtxt(file_sphere4[0])
sphere5 = np.loadtxt(file_sphere5[0])

Nlaser_off = np.loadtxt(path_noise_laser_off[0])
Nlaser_on = np.loadtxt(path_noise_laser_on[0])
# Nlaser_on_lit = np.loadtxt(path_noise_laser_on_lit_open[0])
dig = np.loadtxt(path_digitalization[0])

def func(x,x0,A,d):
    f = A/(x*np.sqrt(d**2 + ((x**2-x0**2)/x)**2))
    return f

fit = np.logical_and(sphere5[0]>40, sphere5[0]<1000)
popt, pcov = curve_fit(func, sphere2[0][fit], sphere2[1][fit], p0 = [220,1e4,0.1])
# popt = [220,1e4,0.1]

#plt.rc('font',family='Times New Roman')

plt.figure()
# plt.loglog(sphere2[0], 20*1e6*func(sphere2[0],*popt)/9.8)

plt.loglog(sphere1[0],20*(sphere1[1]/9.8)*10**6, label = "1mbar no feedback", color = "b")
plt.loglog(sphere2[0],20*(sphere2[1]/9.8)*10**6, label = "1mbar with feedback", color = "orange", alpha = 1)
plt.loglog(sphere3[0],(sphere3[1]/9.8)*10**6, label = "$10^{-6}$mbar with feedback", color = "k", lw = 1)
plt.loglog(sphere4[0], sphere4[1], lw = 1, color = "lawngreen" ,alpha = 1, label = "Laser before chamber")
plt.loglog(sphere4[0], sphere4[1]*func(sphere4[0],*popt)/(5.18e-5), lw = 1, color = "r" ,alpha = 0.6, label = "Predicted noise")

# plt.loglog(Nlaser_off[0],Nlaser_off[1], label = "laser off AC drive")
# plt.loglog(Nlaser_on[0], 20*(Nlaser_on[1])*1e6*0.0038, label = "laser on AC drive")
# plt.loglog(sphere[0],Nlaser_on_lit[1], label = "laser on lits open AC drive")
# plt.loglog(dig[0], dig[1], label = "digitalization")


L2 = plt.legend(loc="center left", bbox_to_anchor=(-0.02,0.59), frameon = False, fontsize = 14, labelspacing = 0.02)
plt.xlabel("Frequency [Hz]", fontsize = 17)
plt.ylabel("$\sqrt{S_a}$ [$\mu g$/$\sqrt{Hz}$]", fontsize = 17)

l = plt.gca().add_patch(
    patches.Rectangle(
        (13,95),   # (x,y)
        85,          # width
        10,          # height
        edgecolor=None,
        facecolor="#ffffff"
))
l.set_zorder(20)
L2.set_zorder(20) 
plt.xlim(10, 1000)
plt.ylim(0.5, 1000)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.grid()
plt.tight_layout(pad = 0)
plt.show()
# plt.savefig("test.png")

# from PIL import Image


 
# img=Image.open('test.png').convert('L')
# arr = np.asarray(img)
# plt.imshow(arr, cmap=pylab.gray())
# plt.show()
