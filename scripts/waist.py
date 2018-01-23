import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import glob
import re
import scipy.special
import scipy.optimize


path = r"C:\data\acceleration_paper\waist_measurement\Y_direction2"

file_list = glob.glob(path+r"\0_*.txt")
file_listerr = glob.glob(path+r"\No_blade.txt")
file_listoff = glob.glob(path+r"\offset.txt")

f = np.loadtxt(file_listerr[0],skiprows=3,usecols = (3,3), unpack = True)[0]
off = np.loadtxt(file_listoff[0],skiprows=3,usecols = (3,3), unpack = True)[0]

# print np.mean(f)
# print np.std(f)


def creat_distance_mean_std_list(file_list):
    mean = []
    dist = []
    std = []
    for i in range(len(file_list)):
        d = float(re.findall("-?\d+in",file_list[i])[0][:-2])
        m = np.mean(np.loadtxt(file_list[i],skiprows=3,usecols = (3,3), unpack = True)[0])
        s = np.std(np.loadtxt(file_list[i],skiprows=3,usecols = (3,3), unpack = True)[0])
        d = 1e-6*d*25.4 # in um
        mean.append(m)
        std.append(s)
        dist.append(d)
    return [dist,mean,std]

d,m,s = creat_distance_mean_std_list(file_list)

m = m + np.mean(off)

merr = np.array(m)*np.std(f)/np.mean(f)

print np.std(f)/np.mean(f)

def gauss(x,A,m,s):
    g = A*np.exp(-((x-m)/s)**2)
    return g

def erf(x,B,A,m,s):
    return B - A*scipy.special.erf((x-m)/(s))

p0 = [1e-4, 6e-5, 4e-5]
popt, pcov = curve_fit(gauss, d[3:8], np.abs(np.gradient(m))[3:8], sigma = merr[3:8], p0 = p0)

# p0 = [0.2, 0.2, 0.2e-4, 5e-5]
# popt, pcov = curve_fit(erf, d[:], m[:], sigma = merr[:], p0 = p0)


dd = np.linspace(d[0],d[-1],1000)

print popt
print np.sqrt(pcov[2][2])

plt.figure()
plt.errorbar(d, np.abs(np.gradient(m)), yerr = merr, fmt = "r--")
plt.plot(dd, gauss(dd,*popt))
# plt.errorbar(d, np.abs(m), yerr = merr, fmt = "ko")
# plt.plot(dd, (erf(dd,*popt)))
# plt.gca().set_yscale("log")
plt.show()

