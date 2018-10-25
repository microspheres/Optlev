import numpy, matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# folder C:\data\20180323\bead3_SiO2_15um_POL_NS\meas26_DC_only

V = (200./1000.)*np.array([0., 429., 857., 1286., 1714., 2143., 2571., 3000.])

rot = np.array([1.384, 1.385, 1.373, 1.354, 1.341, 1.342, 1.350, 1.360])*1.0e6

pre = np.array([0, 0.05, 0.20, 0.47, 0.85, 1.37, 1.96, 2.65])
pre_err = 0.019*np.ones(len(pre))

rho = 1800.
radius = 7.5e-5
Iner = (8./15.)*rho*np.pi*(radius**5)
torque = (Iner*pre*rot)*((2*np.pi)**2)*1.0e18

def curve(z, a, b):
    v, rot = z
    return ((a)*(v**2) + (b)*v)/(Iner*rot)

p0 = np.array([])
Z = [V, rot]
popt, pcov = curve_fit(curve, Z, pre, sigma = pre_err)


vplot = np.linspace(0, 100, 1000)

plt.figure()
for i in range(len(V)):
    label = "$\omega_{rot}$/$2\pi$ = " + str("%0.3E" % rot[i]) + " Hz"
    plt.errorbar(V[i], pre[i], yerr = pre_err[i], fmt = "o", label = label)
plt.plot(V, curve(Z, *popt), "r--", label = "fit")
xlabel = " DC Volt [V]"
ylabel = "$\omega_{p}$/2$\pi$ [Hz]"
plt.ylabel(ylabel)
plt.xlabel(xlabel)
plt.legend()
plt.grid()


torquelin =((popt[1]*V))*((2*np.pi)**2)*1.0e18
torquelin_err = (np.sqrt(pcov[1][1])*V)*((2*np.pi)**2)*1.0e18

torquesq = ((popt[0]*(V**2)))*((2*np.pi)**2)*1.0e18
torquesq_err = (np.sqrt(pcov[0][0])*(V**2))*((2*np.pi)**2)*1.0e18


plt.figure()
label1 = "total torque"
plt.plot(V, torque, "ro", label = label1)
label2 = "permanent dipole torque"
plt.errorbar(V, torquelin, yerr = torquelin_err, fmt = "bo", label = label2)
label3 = "induced dipole torque"
plt.errorbar(V, torquesq, yerr = torquesq_err, fmt = "g*", label = label3)
xlabel = " DC Volt [V]"
ylabel = "Torque [pN.$\mu$m]"
plt.ylabel(ylabel)
plt.xlabel(xlabel)
plt.legend()
plt.grid()

plt.show()














# folder C:\data\20180323\bead3_SiO2_15um_POL_NS\meas24_DC_only

# V = (200./1000.)*np.array([0., 750., 1500., 2250., 3000.])

# rot = np.array([1.272, 1.278, 1.279, 1.285, 1.295])*1.0e6

# pre = np.array([0, 0.17, 0.70, 1.58, 2.78])
# pre_err = 0.019*np.ones(len(pre))

# rho = 1800.
# radius = 7.5e-5
# Iner = (8./15.)*rho*np.pi*(radius**5)
# torque = (Iner*pre*rot)*((2*np.pi)**2)*1.0e18

