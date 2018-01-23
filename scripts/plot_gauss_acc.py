import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob


def gauss(x,a,b,c):
    g = c*np.exp(-0.5*((x-a)/b)**2)
    return g

path = r"C:\data\acceleration_paper\from_dates\20171004bead8_23um_QWP_NS\plot_gaus_hist"

hist = glob.glob(path+"\*.txt") 


data = np.loadtxt(hist[0])

bc = data[0] # in SI units
h = data[1]

bc = bc/9.8 # in g units
bc = bc*1e9 # in nano g units

p0 = [0, 23, 2]

popt, pcov = curve_fit(gauss, bc, h, p0)


space = np.linspace(bc[0]-20,bc[-1]+20, 1000)


print "result from acc fit in nano g"
print popt
print np.sqrt(pcov[0,0])

plt.figure()
plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = 'ro')
plt.plot(space, gauss(space,*popt), 'k')
plt.xlabel("Acceleration [n$g$]", fontsize = 17)
plt.ylabel("Number of measurements", fontsize = 17)
plt.grid()
plt.ylim(0,35)
plt.xlim(-120,120)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.tight_layout(pad = 0)
plt.show()
