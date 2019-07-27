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

path = r'/data/20170530/bead7_15um_QWP/dipole_integralXY/'

file_name = 'ACandDCamplitudes.txt'

distance = 0.002 #m

Vpp_to_Vamp = 0.5

trek = 200.0

F = np.loadtxt(os.path.join(path, file_name))

Ea = trek*Vpp_to_Vamp*F[0]/distance
                       
#g = dE/E

def Fw(Eac, g, p0, back):
    return p0*Eac*g + back

def F2w(Eac, g, alpha, back):
    return alpha*(Eac**2)*g + back

def Fwacdc(Eac, g, p0, back, alpha, Edc):
    return p0*Eac*g + back + alpha*(2.0*g*Edc*Eac)

#plt.figure()
#plt.plot(Ea,F[1],'.')
#plt.plot(Ea,Fw(Ea, *popt_W), 'o')
#plt.plot(Ea,F[2],'.')
#plt.plot(Ea,F2w(Ea, *popt_2W), 'o')
#plt.show()

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

order(Ea, F[1], F[2])

popt_W, pcov_W = curve_fit(Fw, Ea_order, force_W_order)
popt_2W, pcov_2W = curve_fit(F2w, Ea_order, force_2W_order)

plt.figure()
plt.loglog(Ea_order, force_W_order, ".")
plt.loglog(Ea_order, force_2W_order, ".")
plt.loglog(Ea_order, Fw(np.array(Ea_order), *popt_W))
plt.loglog(Ea_order, F2w(np.array(Ea_order), *popt_2W))
plt.show()

def alpha_0(r): # in um
    r1 = 1.0*r/(1e6)
    epsilon0 = 8.854e-12
    return 3.*epsilon0*(2./5.)*(4.*np.pi/3.)*(r1**3)
