import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt
import allantools as al


folder = r"C:\data\20191210\10um\3\newpinhole\acceleration2"
filename = r"\corr_combined.npy"

c = np.load(str(folder)+str(filename))
time = (2**19/1e4)*(np.array(range(len(c)))+1)

def gauss(x,a,b,c):
    g = c*np.exp(-0.5*((x-a)/b)**2)
    return g

def histo(c, bins):
    h,b = np.histogram(c, bins = bins)
    bc = np.diff(b)/2 + b[:-1]
    return [h, bc]


def fit_func(time, A):
    y = A/np.sqrt(time)
    return y


allan = al.totdev(c, rate = 1./52.4288, taus = time)

dev = np.sqrt(allan[1])
err = np.sqrt(allan[2])
x = allan[0]

popt, pcov = opt.curve_fit(fit_func, x, dev, sigma = err)

plt.figure()
plt.errorbar(x, dev, yerr = 0*err, fmt = ".")
plt.loglog(x, fit_func(x, *popt))


# plt.figure()
# plt.plot(c)

M = []
S = []
T = []

for i in range(len(c)):
    if i > 30 and i%10 == 0:
        try:
            bins = 10
            h, bc = histo(c[:i], bins)
            sigma = np.sqrt(h)
            for j in range(len(sigma)):
                if sigma[j] == 0:
                    sigma[j] = 1.
            popt, pcov = opt.curve_fit(gauss, bc, h, sigma = sigma)
            M.append(popt[0])
            S.append(np.sqrt(pcov[0][0]))
            t = (2**19/1e4)*(i+1)
            T.append(t)
        except:
            print "fail"


popt2, pcov2 = opt.curve_fit(fit_func, T, S, sigma = S)


plt.figure()
# plt.errorbar(T, M, yerr = S, fmt = ".")
# plt.ylim(-3,3)
# plt.grid()
plt.loglog(T, S, ".")
plt.loglog(T, fit_func(T, *popt2))

plt.show()

        
