import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

path = r"C:\data\20190202\15um\4\PID"

file1 = r"output_from_feed_backonoff.npy"

file2 = r"Gamma_from_com_temp.npy"

data1 = np.load(os.path.join(path, file1))
data2 = np.load(os.path.join(path, file2))

kb = 1.38*10**-23

mass = 2.*2.3*10**-26

vis = 18.54*10**-6
vis_hidrogen = 1.37e-5

rho = 1800

R = 7.0*10**-6

M = (4./3.)*np.pi*(R**3)*rho

press = 240.

temp = 300

def mean_free_path(vis, press, temp, mass):
    L1 = vis/press
    L2 = np.sqrt( np.pi*kb*temp/(2*mass) )
    return L1*L2

def Kn(vis, press, temp, mass, R):
    L = mean_free_path(vis, press, temp, mass)
    return L/R

def Gamma(vis, press, temp, mass, R, M):
    A = (6.0*np.pi*vis*R/M)
    B = 0.619/(0.619 + Kn(vis, press, temp, mass, R))
    C = (1. + 0.31*Kn(vis, press, temp, mass, R)/(0.785 + 1.152*Kn(vis, press, temp, mass, R) + Kn(vis, press, temp, mass, R)**2))
    return A*B*C

Gammaover2pi_1mbar = Gamma(vis, 1.*100., temp, mass, R, M)
Gammaover2pi_lp = data2[3]
press_mbar_pd = data2[4]


plt.figure()
plt.errorbar(data1[0], data1[1], yerr = data1[2], fmt = "bo", label = "heating from ON OFF data")
plt.errorbar(data1[0], data1[3], yerr = data1[4], fmt = "ro", label = "damping from ON OFF data")
plt.errorbar(data2[0], data2[1], yerr = data2[2], fmt = "kx", label = "damping from PSD fit")
hh = np.arange(0,1,0.01)
plt.plot(hh, Gammaover2pi_1mbar*np.ones(len(hh)), "k--", label = "gas  at 1 mbar")
name = "gas  at " + str("%0.1E" % press_mbar_pd) + " mbar"
plt.plot(hh, Gammaover2pi_lp*np.ones(len(hh)), "b--", label = name)
plt.yscale("log")
plt.xscale("log")
plt.grid()
plt.legend(loc = 3)
plt.ylim(1e-6, 5e2)
plt.xlim(0.01, 1)
plt.ylabel("$\Gamma / 2\pi$ [Hz]")
plt.xlabel("Feedback dgx")
plt.tight_layout(pad = 0)




f, (ax, ax2) = plt.subplots(2, 1, sharex = True)


ax.errorbar(data1[0], data1[1], yerr = data1[2], fmt = "bo", label = "heating from ON OFF data")
ax.errorbar(data1[0], data1[3], yerr = data1[4], fmt = "ro", label = "damping from ON OFF data")
ax.errorbar(data2[0], data2[1], yerr = data2[2], fmt = "kx", label = "damping from PSD fit")
hh = np.arange(0,1,0.01)
ax.plot(hh, Gammaover2pi_1mbar*np.ones(len(hh)), "k--", label = "gas  at 1 mbar")
name = "gas  at " + str("%0.1E" % press_mbar_pd) + " mbar"
ax2.plot(hh, Gammaover2pi_lp*np.ones(len(hh)), "b--", label = name)
ax.set_yscale("log")
ax.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xscale("log")
ax.grid()
ax2.grid()
ax.legend()
ax2.legend()
ax.set_ylim(1e-1, 5e2)
ax2.set_ylim(2e-6, 3e-6)
ax.set_xlim(0.01, 1)
ax2.set_xlim(0.01, 1)
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
ax.set_ylabel("$\Gamma / 2\pi$ [Hz]")
ax2.set_ylabel("$\Gamma / 2\pi$ [Hz]")
plt.xlabel("Feedback dgx")
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
plt.tight_layout(pad = 0)




# plt.figure()
# plt.loglog(np.arange(1e-7, 1, 1e-6), Gamma(vis, np.arange(1e-7, 1, 1e-6)*100., temp, mass, R, M))

plt.show()
