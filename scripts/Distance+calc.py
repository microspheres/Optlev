
# coding: utf-8

# In[13]:

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import lambertw
import matplotlib.pyplot as plt

Vin = 5.0 #volts high Z
E0 = 8.854187E-12
F = 2.0E7
R = .0127
e = 2.718281828

Vout = np.array([3.14, 1.27, 0.84, 0.56, 0.38, 0.28, 0.23, 0.19])

Hexp = Vout/Vin

distance = 0.001*np.array([47.3, 47.4, 47.6, 47.9, 48.1, 48.5, 48.9, 49.2])

relative = distance - np.min(distance)

def H(d, R1, R2, A, d0):
    # SI units
    C = A/(d+d0)
    return R1/np.sqrt(R2**2 + 1./((2.*np.pi*F)*C)**2)

def Hadam(d, R1, R2, d0):
    # SI units
    dist = d+d0
    C = R**2*np.pi*E0/dist + E0*R*np.log(4*np.pi*e*R/dist)
    return R1/np.sqrt(R2**2 + 1./((2.*np.pi*F)*C)**2)



poptadam, pcovadam = curve_fit(Hadam, relative, Hexp, p0=(50, 3, 1.0E-4), bounds=([1.0E-25, 1.0E-25, 1.0E-25], [100., 100., 0.5,]))
popt, pcov = curve_fit(H, relative, Hexp, p0=(35, 1.58, 8.8E-15, 9.1E-05), bounds=([1.0E-25, 1.0E-25, 1.0E-25, 1.0E-25], [100., 100., 1.0E-13, 0.5]))
print(poptadam)
print(popt)
plt.figure()
plt.plot((relative + poptadam[2])*1000, Hexp, "bo")
plt.plot((relative + poptadam[2])*1000, Hadam(relative, *poptadam), "g-")
plt.plot(relative + poptadam[2], H(relative, *popt), "r-")
plt.xlabel("distance (mm)")
plt.ylabel("Transfer function")
plt.title('distance between plates')
plt.grid()
plt.show()

plt.plot((relative+poptadam[2])*1000, Hadam(relative, *poptadam) - Hexp, "g.")
plt.xlabel("distance (mm)")
plt.ylabel("Residual")
plt.grid()
plt.show()

interppoints = np.linspace(0, np.amax(relative)+np.ptp(relative)/4, 100)
Hinterp = Hadam(interppoints, *poptadam)

cf = 191/189 #correction factor due to chamber closing and pressure
V_sphere = 1.25 #.383 in VPP
H_sphere = V_sphere/Vin

def distance(H, R1, R2, A, d0):
    return A*(2*np.pi*F)*np.sqrt(((R1/H)**2 - R2**2))

def distanceadam(H, R1, R2, d0):
    return np.interp(H, np.flipud(Hinterp), np.flipud(interppoints))+poptadam[2]

print("distance in m")
print(distance(H_sphere*cf, *popt))
print(distanceadam(H_sphere*cf, *poptadam))


# In[ ]:


plt.plot(relative+poptadam[2], Hexp, "bo")
plt.plot(interppoints, Hadam(interppoints, *poptadam), "g-")
plt.plot(interppoints, H(interppoints, *popt), "r-")
plt.title('first plot but with more points')
plt.show()




import matplotlib

plt.rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

plt.plot(np.array(relative+poptadam[2])*1000., Hexp, "bo")
plt.plot(np.array(interppoints)*1000., Hadam(interppoints, *poptadam), "g-")

plt.xlabel("Distance between plates [um]", size=22)
plt.ylabel("Vout/Vin", size=22)

#plt.legend(prop={'size': 22}, numpoints=1, loc=2)

plt.grid()
plt.show()
