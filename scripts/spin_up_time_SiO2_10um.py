import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


### Data for SiO2 spheres
P_data = [2.1e-6, 1.8e-7, 5.4E-7]
xerr = 0.3*np.array(P_data)
Tau_data = [2210., 27000.0, 8800.]
yerr = 0.15*np.array(Tau_data)

# dia = 10.3
r = 5.15e-6 # m
rho = 1800.0 # kg/m3
kb = 1.380648e-23 # SI
T = 293.0 # K
m = (18/28.)*4.65e-26 # H2o molecule mass in kg
# m = 4.65e-26 # N2 molecule mass in kg

dr = 0.136*r
drho = 0.*rho

pi = np.pi

def mean_speed(T,m):
    a = np.sqrt(8.0*kb*T/(pi*m)) # mean (arithmetic) speed
    # a = np.sqrt(3.0*kb*T/(m)) # rms speed
    return a

def spinuptime(p,r,rho,T,m): # p is in mbar
    p = p*100.
    a = 0.1*pi*r*rho*(mean_speed(T,m))/p
    return a

def spinuptime_err(p,r,rho,T,m, dr, drho): # p is in mbar
    p = p*100.
    err = (0.1*pi*r*rho*(mean_speed(T,m))/p)*np.sqrt( (dr/r)**2 + (drho/rho)**2 )
    return err

def linear(x, a, b):
    return a/x + b

print (mean_speed(T,m))

P = np.arange(1e-7, 4e-5, 1e-8)

Tau = spinuptime(P,r,rho,T,m)

Tau_err = spinuptime_err(P,r,rho,T,m, dr, drho)

popt, pcov = curve_fit(linear, Tau_data, P_data)


plt.figure()
### SiO2 10um
plt.loglog(P, Tau, label = "Model SiO2 spheres")
plt.fill_between(P, Tau + Tau_err, Tau - Tau_err, alpha = 0.5)
plt.errorbar(P_data, Tau_data, xerr=xerr, yerr = yerr, fmt='ro')
plt.plot(P, linear(P, *popt), "k--", label = "Fit")

### vaterite



plt.legend(loc=0)
plt.ylabel("Damping time [s]", fontsize = 16)
plt.xlabel("Pressure [mbar]", fontsize = 16)
plt.rcParams.update({'font.size': 14})
plt.tight_layout(pad = 0)
plt.grid()
plt.show()
