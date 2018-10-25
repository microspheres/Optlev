import numpy as np
import matplotlib.pyplot as plt

contractor = True

#fixed
l = 1.064E-6
NA0 = 0.14
H0 = 1.7E-4

#beam contraction
f1 = 105E-2
f2 = 25E-2

#free
FF= 25E-2 # parabolic
F0 = 25E-2

Z = 1E-3
Z1 = Z/2

a = 0.5 # size intensity


if contractor == False:
    def NAF(NA0, ff, f0):
        return 1.0*NA0*f0/ff

    def wF(l, NA0, ff, f0):
        return 1.0*l/(np.pi*NAF(NA0, ff, f0))
    
    def HF(H0, ff, f0):
        return 1.0*H0*ff/f0
        
    def beam_W(z, l, NA0, ff, f0):
        zr = 1.0*(wF(l, NA0, ff, f0)**2)*np.pi/l
        return  wF(l, NA0, ff, f0)*np.sqrt(1 + (z/zr)**2)

    def Intensity(z, r, l, NA0, ff, f0):
        return 1.0*((wF(l, NA0, ff, f0)/beam_W(z, l, NA0, ff, f0))**2)*np.exp(-2*(r/beam_W(z, l, NA0, ff, f0))**2)

    def patern(H0, ff, f0, z, r, l, NA0, Pl, Pr):
        aux = Intensity(z, r, l, NA0, ff, f0) + Pr*Intensity(z, r - HF(H0, ff, f0), l, NA0, ff, f0) + Pl*Intensity(z, r + HF(H0, ff, f0), l, NA0, ff, f0)
        return aux

    x = np.linspace(-HF(H0, FF, F0), HF(H0, FF, F0), 4000)

    print "NA", NAF(NA0, FF, F0)
    print "waist in um", beam_W(Z1, l, NA0, FF, F0)*1E6
    print "height in mm", Z1*1000
    
    plt.figure()
    plt.plot(x, patern(H0, FF, F0, Z1, x, l, NA0, 0.0, a), label = Z1)
    plt.plot(x, patern(H0, FF, F0, Z1, x, l, NA0, a, 0.0))
    plt.plot(x, Intensity(Z1, x, l, NA0, FF, F0), "k--",label = "original")
    plt.legend()
    plt.show()
    



if contractor == True:
    def NAF(NA0, ff, f0, f1, f2):
        return 1.0*NA0*(f0/ff)*(f2/f1)

    def wF(l, NA0, ff, f0, f1, f2):
        return 1.0*l/(np.pi*NAF(NA0, ff, f0, f1, f2))
    
    def HF(H0, ff, f0, f1, f2):
        return 1.0*H0*(ff/f0)*(f1/f2)

    def beam_W(z, l, NA0, ff, f0, f1, f2):
        zr = 1.0*(wF(l, NA0, ff, f0, f1, f2)**2)*np.pi/l
        return  wF(l, NA0, ff, f0, f1, f2)*np.sqrt(1 + (z/zr)**2)

    def Intensity(z, r, l, NA0, ff, f0, f1, f2):
        return 1.0*((wF(l, NA0, ff, f0, f1, f2)/beam_W(z, l, NA0, ff, f0, f1, f2))**2)*np.exp(-2*(r/beam_W(z, l, NA0, ff, f0, f1, f2))**2)

    def patern(H0, ff, f0, f1, f2, z, r, l, NA0, Pl, Pr):
        aux = Intensity(z, r, l, NA0, ff, f0, f1, f2) + Pr*Intensity(z, r - HF(H0, ff, f0, f1, f2), l, NA0, ff, f0, f1, f2) + Pl*Intensity(z, r + HF(H0, ff, f0, f1, f2), l, NA0, ff, f0, f1, f2)
        return aux
        
    x = np.linspace(-HF(H0, FF, F0, f1, f2), HF(H0, FF, F0, f1, f2), 4000)
        
    print "NA", NAF(NA0, FF, F0, f1, f2)
    print "waist in um", beam_W(Z1, l, NA0, FF, F0, f1, f2)*1E6
    print "height in mm", Z1*1000
    
    plt.figure()
    plt.plot(x, patern(H0, FF, F0, f1, f2, Z1, x, l, NA0, 0.0, a), label = Z1)
    plt.plot(x, patern(H0, FF, F0, f1, f2, Z1, x, l, NA0, a, 0.0))
    plt.plot(x, Intensity(Z1, x, l, NA0, FF, F0, f1, f2), "k--",label = "original")
    plt.legend()
    plt.show()
