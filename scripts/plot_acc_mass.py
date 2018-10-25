import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from scipy.optimize import curve_fit

def f1(x,a):
    f = a*1./np.sqrt(x)
    return f

def f2(x,a):
    f = a*1./x
    return f

m = np.array([11.8, 10.2, 2.9, 2.7, 2.2, 1.3, 0.12])

merr = m*0.17

acc = ((1.0*10**-5)*np.array([0.38, 0.65, 0.88, 1.20, 1.18, 2.5, 3.6])/9.8)*1.e6 # in ug

accerr = acc*0.18

# yaxis = [0.5, 1.0, 2.0, 4.0]
yaxis = [0.1,1,10]

popt, pcov = curve_fit(f1, m, acc, sigma = accerr)
popt2, pcov2 = curve_fit(f2, m, acc, sigma = accerr)

print "for 1/sqrt(mass)"
print popt
print np.sqrt(pcov)
print "for 1/mass"
print popt2
print np.sqrt(pcov2)

plt.figure()
plt.errorbar(m, acc, xerr=merr, yerr = accerr, fmt='ro')
plt.plot(m, f1(m,*popt), "r--", label = "$\propto$ Mass$^{-1/2}$")
# plt.plot(m, f2(m,*popt), "k:", label = "$\propto$ Mass$^{-1}$")
plt.yscale('log')
plt.xscale('log')
plt.xlabel("Mass [ng]", fontsize = 17)
plt.ylabel("Acceleration Sensitivity [$\mu$g/$\sqrt{Hz}$]", fontsize = 17)
plt.yticks(yaxis, fontsize=16)
plt.xticks(yaxis, fontsize=16)
plt.gca().set_yticklabels(yaxis)
plt.gca().set_xticklabels(yaxis)
plt.xticks(fontsize=15)
# plt.grid(which = "minor")
plt.ylim(0.1,10)
plt.grid(which = "major")
# plt.subplots_adjust(bottom = 0.15)
plt.tight_layout(pad = 0)
plt.legend(fontsize=16)


plt.figure()
plt.errorbar(m, acc*np.sqrt(m), xerr=merr, yerr = acc*np.sqrt(m)*0.25, fmt='ro')
plt.xlabel("Mass [ng]", fontsize = 17)
plt.ylabel("$\sqrt{S}$ [$\mu$g/($\sqrt{Hz}\sqrt{m^{-1}}$)]", fontsize = 17)
plt.yscale('log')
plt.xscale('log')
plt.tight_layout(pad = 0)
plt.show()
