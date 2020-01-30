import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
from scipy.special import wofz
import numpy as np
import glob
import scipy.optimize as opt
from scipy.stats import levy_stable
import matplotlib.cm as cm

# [freqhp, freq_meas, xpsd2_m_hp, xpsd2_m_meas, xpsd2_v_hp, xpsd2_meas_v, xDg, fres]

pi = np.pi
folder = r"C:\data\20191122\10um\2\nofeedback6"


name_load_out = str(folder) + "\info_outloop.npy"
dataout = np.load(name_load_out)

freq_meas = dataout[1]
xpsd2out = dataout[5]

name_load_in = str(folder) + "\info_inloop.npy"
datain = np.load(name_load_in)

freq_meas = datain[1]
xpsd2in = datain[5]

def psd3(f,A,f0,gam,sig):

    omega = 2*np.pi*f
    omega_0 = 2*np.pi*f0

    z = ((omega**2 - omega_0**2) + 1j * omega*gam)/(np.sqrt(2)*sig)

    V = np.abs(A*np.real( wofz(z) )/sig)

    return np.sqrt(V)

def psd3_2(f,A,fx,fy,gam,sig):

    omega = 2*np.pi*f
    omega_x = 2*np.pi*fx
    omega_y = 2*np.pi*fy

    zx = ((omega**2 - omega_x**2) + 1j * omega*gam)/(np.sqrt(2)*sig)
    zy = ((omega**2 - omega_y**2) + 1j * omega*gam)/(np.sqrt(2)*sig)

    V = np.abs(A*np.real( wofz(zx) )/sig) + np.abs(A*np.real( wofz(zy) )/sig)

    return np.sqrt(V)

fit_points1 = np.logical_and(freq_meas > 53, freq_meas < 59.8)
fit_points2 = np.logical_and(freq_meas > 60.1, freq_meas < 68)

fit_points = fit_points1 + fit_points2 


poptQ, pcovQ = opt.curve_fit(psd3_2, freq_meas[fit_points], np.sqrt(xpsd2in[fit_points]), p0 = [3.4e2, 6.1e+01, 6.12e+01, 1.8e-02, 7e+01], sigma = 0.1*np.sqrt(xpsd2in[fit_points]))
print poptQ
print np.sqrt(pcovQ[2][2])

poptQ2, pcovQ2 = opt.curve_fit(psd3, freq_meas[fit_points], np.sqrt(xpsd2in[fit_points]), p0 = [3.4e2, 6.12e+01, 1.8e-02, 7e+01], sigma = 0.1*np.sqrt(xpsd2in[fit_points]))
print poptQ2
print np.sqrt(pcovQ2[2][2])

# fit_points0z = np.logical_and(freq_meas > 6, freq_meas < 7.2)
fit_points1z = np.logical_and(freq_meas > 8.5, freq_meas < 10.8)
fit_points2z = np.logical_and(freq_meas > 11.5, freq_meas < 14)

fit_pointsz = fit_points1z + fit_points2z

poptQ2z, pcovQ2z = opt.curve_fit(psd3, freq_meas[fit_pointsz], np.sqrt(xpsd2in[fit_pointsz]), p0 = [1.4e1, 10.36, 1.8e-02, 2e+01], sigma = 0.1*np.sqrt(xpsd2in[fit_pointsz]))
print poptQ2z
print np.sqrt(pcovQ2z[2][2])


f = np.linspace(53, 68, 3000)
fz = np.linspace(8.5, 14, 3000)

fig  = plt.figure()
plt.rcParams.update({'font.size': 10})
plt.scatter(freq_meas, np.sqrt(xpsd2in), marker = ".", s = 3, color = "C0")
plt.loglog(f, psd3(f, *poptQ2), "-", alpha = 0.95, color = "C1")
plt.loglog(fz, psd3(fz, *poptQ2z), "-", alpha = 0.95, color = "C3")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency [Hz]")
plt.ylabel("$\sqrt{S_{xx}}$ [V/$\sqrt{Hz}$]")
plt.xlim(1,200)
plt.ylim(1e-4, 10)
plt.grid()
plt.tight_layout(pad = 0)
fig.set_size_inches(4,2.5)
plt.show()
