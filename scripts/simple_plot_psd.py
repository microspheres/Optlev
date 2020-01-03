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

plt.figure(figsize=(5,3))
plt.rcParams.update({'font.size': 14})
plt.scatter(freq_meas, np.sqrt(xpsd2in), marker = ".", s = 6)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency [Hz]")
plt.ylabel("$\sqrt{S_{xx}}$ [volts/$\sqrt{Hz}$]")
plt.xlim(1,200)
plt.ylim(1e-4, 10)
plt.grid()
plt.tight_layout(pad = 0)
plt.show()
