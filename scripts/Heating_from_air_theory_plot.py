import matplotlib.pyplot as plt
import numpy as np

# from page 125 of tongcang thesis:

kb = 1.38e-23

mass_H2 = 3.35e-27 # Kg
mass_N2 = 4.65e-26 # Kg

vis_air = 2.98e-5
vis_N2 = 2.86e-5
vis_H2 = 1.37e-5 # Pa*s, see https://www.engineeringtoolbox.com/gases-absolute-dynamic-viscosity-d_1888.html

rho = 1800.0 #kg/m^3

R = 7.5e-6 #m

M = (4./3.)*np.pi*(R**3)*rho

temp = 300. #K

def mean_free_path(vis, press, temp, mass):
    L1 = vis/press
    L2 = np.sqrt( np.pi*kb*temp/(2*mass) )
    return L1*L2

def Kn(vis, press, temp, mass, R):
    L = mean_free_path(vis, press, temp, mass)
    return L/R

def Gamma(vis, press, temp, mass, R, M):
    A = (6.0*np.pi*vis*R/M)
    B = 0.619/(0.619 + Kn(vis, press, temp, mass, R))
    C = (1. + 0.31*Kn(vis, press, temp, mass, R)/(0.785 + 1.152*Kn(vis, press, temp, mass, R) + Kn(vis, press, temp, mass, R)**2) )
    return A*B*C

# pressure for the function is in Pa

P_H2 = np.arange(1, 10)
P_H2 = P_H2**2 / 1e5
Pmbar_H2 = P_H2/100.

P_N2 = np.arange(1, 3000)
P_N2 = P_N2**2 / 1e3
Pmbar_N2 = P_N2/100.

Gamma_H2 =  Gamma(vis_H2, P_H2, temp, mass_H2, R, M)
Gamma_N2 =  Gamma(vis_N2, P_N2, temp, mass_N2, R, M)

plt.figure()
axis_font = {'size':'16'}
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 
plt.loglog(Pmbar_H2, Gamma_H2/(2*np.pi), "r-", label = "H2")
plt.loglog(Pmbar_N2, Gamma_N2/(2*np.pi), "b-", label = "N2")
plt.hlines(2.6, 1e-7, 100, colors='k', linestyles='--', label = "Measured at low pressure")
plt.xlabel("Pressure [mbar]", **axis_font)
plt.ylabel(r"$\Gamma / 2 \pi$ [Hz]", **axis_font)
plt.legend()
plt.tight_layout(pad = 0)
plt.grid()
plt.show()
