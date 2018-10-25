import numpy as np
import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import scipy.signal as sp
import glob
from scipy.optimize import curve_fit
import re


Fs = 1e4

NFFT = 2**12

Volts = False # use true for PSD in volts, use false for PSD in mW.


path = r"C:\data\cobolt_20180727"

file_list = glob.glob(path+"\*.txt")


def plot_psd(l):
    A = []
    Power = []
    for i in l:
        a = np.loadtxt(i)
        A.append(a)
        n = re.findall(r"[-+]?\d*\.\d+|\d+", i)[-1]
        n = float(n)
        Power.append(n)

    if not Volts:
        #calibration
        AM = []
        for i in range(len(A)):
            am = np.mean(A[i])
            AM.append(am)
    
            def line(x, a, b):
                return a*x + b
                
        popt, pcov = curve_fit(line, AM, Power)

        B = []
        for i in range(len(A)):
            A[i] = line(A[i], *popt)
            B.append(np.mean(A[i]))

        V = np.linspace(min(AM), max(AM), 100)
        plt.figure()
        plt.plot(AM, B, "ro")
        plt.plot(V, line(V, *popt), "k--")
        plt.xlabel("Volt [V]")
        plt.ylabel("Power (in the powermeter) [mW]")
        plt.grid()
        plt.tight_layout(pad = 0)


    PSD = []
    Freq = []
    for i in range(len(A)):
        psd_aux, freqs_aux = matplotlib.mlab.psd(A[i] - np.mean(A[i]), Fs = Fs, NFFT = NFFT)
        PSD.append(psd_aux)
        Freq.append(freqs_aux)
    
    plt.figure()
    for i in range(len(A)):
        name = str(Power[i]) + 'mW'
        plt.loglog(Freq[i], np.sqrt(PSD[i]), label = name)
    plt.xlabel("Frequency [Hz]")
    if not Volts:
        plt.ylabel("PSD [mW]" + "$/\sqrt{Hz}$")
    else:
        plt.ylabel("Volts at powermeter [V]")
    plt.title("Laser at 2W attenuated by HWP+PBS")
    plt.grid()
    plt.legend()
    plt.tight_layout(pad = 0)
    
    return [PSD, Freq]
    
plot_psd(file_list)
plt.show()
