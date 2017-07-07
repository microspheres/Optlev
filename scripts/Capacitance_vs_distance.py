import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

Vin = 5.0 #volts high Z
F = 2.0E7

Vout = np.array([3.14, 1.27, 0.84, 0.56, 0.38, 0.28, 0.23, 0.19])

Hexp = Vout/Vin

distance = 0.001*np.array([47.3, 47.4, 47.6, 47.9, 48.1, 48.5, 48.9, 49.2])

relative = distance - np.min(distance)

def H(d, R1, R2, A, d0):
    # SI units
    C = A/(d+d0)
    return R1/np.sqrt(R2**2 + 1./((2.*np.pi*F)*C)**2.)

popt, pcov = curve_fit(H, relative, Hexp, bounds=(1E-20, [100., 100., 1.0E-13, 0.5]))

print popt
print pcov

# plt.figure()
# plt.plot(1000.*(relative + popt[3]), Hexp, "o")
# plt.plot(1000.*(relative + popt[3]), H(relative, *popt), "-")
# plt.xlabel("distance (mm)")
# plt.ylabel("Transfer function")
# plt.grid()
# plt.show()

cf = 191./189. #correction factor due to chamber closing and pressure

V_sphere = 0.315# in VPP

H_sphere = V_sphere/Vin

def distance(H, R1, R2, A, d0):
    return A*(2*np.pi*F)*np.sqrt(((R1/H)**2 - R2**2))

print "distance in m"
print distance(H_sphere*cf, *popt)
