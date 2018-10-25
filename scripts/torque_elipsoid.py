import numpy as np

## torque units are in cgs, then the torque is in dyne*cm.

pi = np.pi

epi = 5.

R = 7.5e-4 # cgs (cm)

Vol = (4./3.)*pi*(R**3)

d_cgs = 0.3 # in cm

V_si = 600. # V

c_cgs = 2.99792458e10 # cm/s

V_cgs = (V_si*1.0e8)/c_cgs # in statV

E = V_cgs/d_cgs

def torque(E, n, angle):
    aux1 = (epi - 1.0)**2
    aux2 = np.abs(1.0 - 3.0*n)
    aux3 = E**2
    aux4 = Vol
    aux5 = np.sin(2.0*angle)
    aux6 = (n*epi + 1 - n)
    aux7 = ((1-n)*epi + 1.0 + n)
    t = aux1*aux2*aux3*aux4*aux5/(aux6*aux7)
    return t
    
print torque(E, 0.31, pi/4)
