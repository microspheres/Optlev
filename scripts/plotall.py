import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import glob
import scipy.optimize as opt

# freqhp, freq_meas, xpsd2_m_hp, xpsd2_m_meas, xpsd2_v_hp, xpsd2_meas_v, xDg


folder_list = [r"C:\data\paper3\22um\PreChamber_LP\1\temp_x\1", r"C:\data\paper3\22um\PreChamber_LP\1\temp_x\11",]

folder_nosphere = r"C:\data\paper3\22um\PreChamber_LP\1\nosphere"



freqmin = 10.
freqmax = 300.
N = 2**13

def chunk(timestream, number):
    l = len(timestream)
    n = int(l/number)
    a = 0
    New = []
    Time = []
    for i in range(number):
        t = np.mean(timestream[0+a:n+a-1])
        New.append(t)
        Time.append(a + int(n/2))
        a = a + n
    return [np.array(New), np.array(Time)]

def HP_data(folder):
    name_load = str(folder) + "\info.npy"
    data = np.load(name_load)
    freq = data[0]
    x = np.sqrt(data[2])
    return [freq, x]

def LP_data(folder_list):
    A = []
    for i in folder_list:
        B = []
        name_load = str(i) + "\info.npy"
        data = np.load(name_load)
        freq = data[1]
        x = np.sqrt(data[3])
        B.append(freq)
        B.append(x)
        A.append(B)
    return A

def nosphere_data(folder):
    name_load = str(folder) + "\info.npy"
    data = np.load(name_load)
    freq = data[1]
    x = np.sqrt(data[3])
    return [freq, x]

plt.figure()

dataHP = HP_data(folder_list[0])
plt.loglog(chunk(dataHP[0], N)[0], chunk(dataHP[1], N)[0])

datanosphere = nosphere_data(folder_nosphere)
plt.loglog(chunk(datanosphere[0], N)[0], chunk(datanosphere[1], N)[0])

data_meas = LP_data(folder_list)
for i in range(len(folder_list)):
    plt.loglog(chunk(data_meas[i][0], N)[0], chunk(data_meas[i][1], N)[0])

plt.xlim(freqmin, freqmax)
plt.ylim(2e-12, 3e-8)
plt.grid()

plt.show()
