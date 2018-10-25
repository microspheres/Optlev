import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt


# using eqs 4.32 to 4.35 as approximation

def n(e): # is is the excentricty in the case a > b = c and e is approx sqrt(2deltaR/R)
    nz = (1./3.) + (2./15.)*e*e
    nx = (1./3.) - (2./15.)*e*e
    ny = (1./3.) - (2./15.)*e*e
    return [nz,nx,ny]


def torque(n,eps,alpha,E,R): # CGS units
    n = n[0]
    V = (4./3.)*np.pi*(R**3)
    t1 = ((eps - 1.0)**2)*(1-3*n)*E*E*V*np.sin(2*alpha)
    t2 = 8*np.pi*(n*eps + 1 - n)*((1-n)*eps + 1 + n)
    t = t1/t2
    return t


Vsi = 200.
w = 2.0*np.pi*355. #Hz and is due to the light polarization

dsi = 0.001
e = 0.04
alpha = np.pi/4.
alpha_guess = np.pi/2. - np.arctan(n(e)[1]/n(e)[0])

Rcgs = 7.5*(10**-4) # cm
Rsi = (7.5*10**-6) # SI
E = (Vsi/dsi)*((10**4)/(299792458.)) # is the field in CGS units
eps = 3.7 # of glass, it can be higher...

torquesi = torque(n(e),eps,alpha_guess,E,Rcgs)*(10**-7) # SI units

msi = (4./3.)*np.pi*((Rsi)**3)*(2000.) # in SI
Isi = 2./5.*msi*(Rsi**2)

wp = torquesi/(Isi*w)
fp = (torquesi/(Isi*w))/(2*np.pi)

print "precession frequency [Hz]"
print fp

# the idea of div by 5000 is that there are 5000 freq bins and the AC field makes the frequency sweep and the width of the AC is 1 bin.
print "effective force [N]"
print ((e*e*Rsi/2.)*msi*(2.*np.pi*230.)**2)*(1./5000.)*np.sin(alpha_guess)

electron = 1.6*10**-19
print "effective charge"
print (((e*e*Rsi/2.)*msi*(2.*np.pi*230.)**2)/((electron*Vsi/dsi)*(1.*10**15)))*(1./5000.)*np.sin(alpha_guess)


# polarization tensor pg 24 where x,y,z are sphere axis (NOT xyz of the measurement!!!)

tzz = n(e)[0]*(4./3.)*np.pi*(Rsi**3)/(4.*np.pi)
txx = n(e)[1]*(4./3.)*np.pi*(Rsi**3)/(4.*np.pi)
tyy = n(e)[2]*(4./3.)*np.pi*(Rsi**3)/(4.*np.pi)

print "tensor components SI units for small eccentricty"
print [tzz,txx,tyy]
