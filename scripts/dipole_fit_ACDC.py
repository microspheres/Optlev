#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:49:30 2017
@author: fernandomonteiro
"""

import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab

path = r'C:\data\20170511\bead2_15um_QWP\new_sensor_feedback\charge16_piezo_56.9_74.9_75.4'

file_name = 'ACandDCamplitudes.txt'


distance = 0.002 #m

Vpp_to_Vamp = 0.5

trek = 200.0

F = np.loadtxt(os.path.join(path, file_name))

Ea = trek*Vpp_to_Vamp*F[0]/distance

Ed = 0*trek*Vpp_to_Vamp*F[3]/distance
         


              
#g = dE/E

def Fw(Eac, g, p0, back):
    return p0*Eac*g + back

def F2w(Eac, g, alpha, back):
    return alpha*(Eac**2)*g + back

def Fwacdc(X, g, back):
    Eac, Edc, alpha, p0 = X
    return p0*Eac*g + back + alpha*(2.0*g*Edc*Eac)



def alpha_0(r): # in um
    r1 = 1.0*r/(1e6)
    epsilon0 = 8.854e-12
    return 3.*epsilon0*(2./5.)*(4.*np.pi/3.)*(r1**3)





def getmin_index(A):
    a = np.min(A)
    b = 0
    for i in np.arange(len(A)):
        if A[i] == a:
            b = i
    return b            


Ea_order = []
force_W_order = []
force_2W_order = []

def order(A,B,C):
    x = A
    y = B
    z = C
    im = getmin_index(A)
    s1 = x[im]
    s2 = y[im]
    s3 = z[im]
    Ea_order.append(s1)
    force_W_order.append(s2)
    force_2W_order.append(s3)
    x = np.delete(x,im)
    y = np.delete(y,im)
    z = np.delete(z,im)
    
    if len(x) > 0:
        order(x,y,z)
    else:
        return Ea_order, force_W_order, force_2W_order


alpha0 = np.ones(len(Ea))*alpha_0(7.5)
p0 = 7.1e-22
p0 = np.ones(len(Ea))*p0

order(Ea, F[1], F[2])


# popt_W, pcov_W = curve_fit(Fw, Ea_order, force_W_order)
popt_2W, pcov_2W = curve_fit(F2w, Ea_order, force_2W_order)

popt_W_dc, pcov_W_dc = curve_fit(Fwacdc, (Ea_order ,Ed , alpha0, p0), force_W_order)

plt.figure()
plt.plot(Ea_order, force_W_order, ".")
plt.plot(Ea_order, force_2W_order, ".")
# plt.plot(Ea_order, Fw(np.array(Ea_order), *popt_W))
plt.plot(Ea_order, F2w(np.array(Ea_order), *popt_2W))

Y = np.array(Ea_order), np.array(Ed), np.array(alpha0), np.array(p0)

plt.plot(Ea_order, Fwacdc(Y, *popt_W_dc))

plt.ylabel("Force (N)")
plt.xlabel("AC field amplitude (N/e)")
plt.show()



print 'alpha ='
print alpha_0(7.5)
print 'p0 ='
print p0[0]
print 'g ='
print popt_W_dc[0]
