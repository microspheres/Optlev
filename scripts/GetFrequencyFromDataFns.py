import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
from PlotPowerSpectrum import getdata

f15um = r"15um_1mbar_nocool_G200.h5"
f5um = r"5um_4mbar_nocool.h5"
spath = r"C:\Users\Sumita\Documents\Microspheres\5 and 15 micron test data"

T = 300
Kb = 1.38 * (1e-23)
M = 2100 * ((7.5e-6) ** 3) * (4. / 3) * np.pi

def PSD2(w, a, W, damp):
    return a * (damp / ((W ** 2 - (w) ** 2) ** 2 + (w * damp) ** 2))

def getXandYdata(path, fname0):
    data0 = getdata(os.path.join(path, fname0))
    xdata = 2 * np.pi * data0[0][20:1000]
    ydata = data0[4][20:1000]
    popt, pcov = curve_fit(PSD2, xdata, ydata)
    return xdata, ydata, popt

def getConvPopt(path, fname0):
    xdata, ydata, popt = getXandYdata(path, fname0)
    Conv = popt[0] / ((2 * Kb * T) / M)
    return Conv, popt

def plotFittedPowerSpectrum(path, fname0):
    xdata, ydata, popt = getXandYdata(path, fname0)
    plt.loglog(xdata, PSD2(xdata, *popt), 'g--', label='')
    plt.loglog(xdata, ydata, label="data")
    plt.show()

def fittedResonantFrequency(path, fname0):
    xdata, ydata, popt = getXandYdata(path, fname0)
    return popt[1]
