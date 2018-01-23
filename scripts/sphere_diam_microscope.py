import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import re
import scipy.optimize


path = r"C:\data\acceleration_paper\diameter_20171027\diameter_20171027\23um_20171109\23um_20171109"

path2 = r"C:\data\acceleration_paper\diameter_20171027\diameter_20171027\23um"

file_list = glob.glob(path+r"\*.txt")

file_list2 = glob.glob(path2+r"\*.txt")

file_list = file_list + file_list2


Per = []
for i in file_list:
   a = np.loadtxt(i,skiprows=1,usecols = (5,5), unpack = True)[0]
   Per.append(a)

Perimeter =  np.concatenate(Per)/np.pi
print len(Perimeter)

def gauss(x,a,b,c):
    g = c*np.exp(-0.5*((x-a)/b)**2)
    return g

h,b = np.histogram(Perimeter, bins = 18)

bc = np.diff(b)/2 + b[:-1]

space = np.linspace(bc[0],bc[-1], 1000)

p0 = [23., 1, 1]

popt, pcov = curve_fit(gauss, bc, h, p0)

print popt
print np.sqrt(pcov)

plt.figure()
plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = 'ko')
plt.plot(space, gauss(space,*popt))
plt.grid()
plt.show()
