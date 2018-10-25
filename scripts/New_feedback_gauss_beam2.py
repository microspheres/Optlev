import numpy as np
import matplotlib.pyplot as plt

# labbook pg 97

#fixed
L = 1.064E-6
NA0 = 0.14 # NA of the fiber
H0 = 1.7E-4 # distance between the fibers in m
# H0 = 1.0E-5 # distance between the fibers in m

#beam contraction/expansion
F1 = 100E-2
F2 = 15E-2

#focal
FF = 50E-2 # parabolic
F0 = 50E-2 # lens after fiber

#height above the focal point
Z = 1E-3
Z1 = 3*Z # used on the plots and prints

A = 0.5 # side intensity

def waist_0(na0, l):
    return 1.0*l/(na0*np.pi)

def waist_F(ff, f0, f1, f2, l, na0):
    return 1.0*(ff/f0)*(f1/f2)*waist_0(na0, l)

def NAF(ff, f0, f1, f2, na0, l):
    return 1.0*(f0/ff)*(f2/f1)*na0

def HF(ff, f0, f1, f2, h0):
    return 1.0*(ff/f0)*(f1/f2)*h0

def ZrF(ff, f0, f1, f2, l, na0): # Rayleigh range
    return 1.0*np.pi*((waist_F(ff, f0, f1, f2, l, na0))**2)/l

def beam_wF(z, ff, f0, f1, f2, l, na0):
    return 1.0*waist_F(ff, f0, f1, f2, l, na0)*np.sqrt(1 + (z/ZrF(ff, f0, f1, f2, l, na0))**2)

def Intensity(r, z, ff, f0, f1, f2, l, na0):
    aux = 1.0*((waist_F(ff, f0, f1, f2, l, na0)/beam_wF(z, ff, f0, f1, f2, l, na0))**2)
    aux2 = 1.0*np.exp(-2.0*(r/beam_wF(z, ff, f0, f1, f2, l, na0))**2)
    return aux*aux2

def Result(r, z, ff, f0, f1, f2, l, na0, h0, al, ar):
    auxl = al*Intensity(r + HF(ff, f0, f1, f2, h0), z, ff, f0, f1, f2, l, na0)
    aux = Intensity(r, z, ff, f0, f1, f2, l, na0)
    auxr = ar*Intensity(r - HF(ff, f0, f1, f2, h0), z, ff, f0, f1, f2, l, na0)
    return auxl + aux + auxr



R = np.linspace(-HF(FF, F0, F1, F2, H0), +HF(FF, F0, F1, F2, H0), 2000)

print "NAF", NAF(FF, F0, F1, F2, NA0, L)
print "waistF in um", beam_wF(Z1, FF, F0, F1, F2, L, NA0)*1E6
print "heigth in mm", Z1*1000
print "distance between beams after optics", HF(FF, F0, F1, F2, H0)
print "cont", beam_wF(0, FF, F0, F1, F2, L, NA0)/HF(FF, F0, F1, F2, H0)

plt.figure()
plt.plot(R, Result(R, Z1, FF, F0, F1, F2, L, NA0, H0, A, 0), label = Z1)
plt.plot(R, Result(R, Z1, FF, F0, F1, F2, L, NA0, H0, 0, A))
plt.plot(R, Intensity(R, Z1, FF, F0, F1, F2, L, NA0), "k--", label = "original")
plt.legend()
plt.show()
