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

path = '/data/20170511/bead2_15um_QWP/new_sensor_feedback/charge43_whole_points/60.0_74.9_75.4'

file_name = 'ACamplitudes.txt'




distance = 0.002 #m

Vpp_to_Vamp = 0.5

trek = 200.0 # gain of the trek

F = np.loadtxt(os.path.join(path, file_name))

Ea = trek*Vpp_to_Vamp*F[0]/distance

# Ed =  trek*Vpp_to_Vamp*0.1/distance
         


              
#g = dE/E

def Fw(X, p0, back):
    """fit dipole ac field only at freq w"""
    Eac, g = X
    return p0*Eac*g + back

def F2w(X, g, back):
    """fit dipole ac field only at freq 2w"""
    Eac, alpha = X
    return alpha*(Eac**2)*g + back

def Fwacdc(X, g, p0, back, alpha):
    """fit dipole ac and dc field only at freq w"""
    Eac, Edc = X
    return p0*Eac*g + back + alpha*(2.0*g*Edc*Eac)



def alpha_0(r): # in um
    """alpha0 , r is the radius in um"""
    r1 = 1.0*r/(1e6)
    epsilon0 = 8.854e-12
    return 3.*epsilon0*(2./5.)*(4.*np.pi/3.)*(r1**3)      


def order(A,B,C):
    return zip(*sorted(zip(A,B,C)))


alpha0 = np.ones(len(Ea))*alpha_0(7.5)

Ea_order, force_W_order, force_2W_order = order(Ea, F[1], F[2])

popt_2W, pcov_2W = curve_fit(F2w, (Ea_order, alpha0), force_2W_order)

g_from_fit = np.ones(len(Ea))*popt_2W[0]

popt_W, pcov_W = curve_fit(Fw, (Ea_order, g_from_fit), force_W_order)


print 'alpha0 = '
print alpha_0(7.5)
print 'g = '
print popt_2W[0]
print 'error g = '
print np.sqrt(pcov_2W[0][0])
print 'p0 = '
print popt_W[0]
print 'background (N) = '
print popt_W[1]


plt.figure()
plt.plot(Ea_order, force_W_order, ".")
plt.plot(Ea_order, force_2W_order, ".")
plt.plot(Ea_order, Fw((np.array(Ea_order),np.array(g_from_fit)), *popt_W))
plt.plot(Ea_order, F2w((np.array(Ea_order),np.array(alpha0)), *popt_2W))

plt.ylabel("Force (N)")
plt.xlabel("AC field amplitude (N/e)")
plt.title(path[path.rfind('\\'):])
plt.show()

