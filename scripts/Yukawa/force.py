import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import integrate


G = 6.67408e-11

pi = np.pi


def Surface_constant_R(r, r_min, D): # r is the distance, D is sphere diameter

    ra = r_min

    h1 = 1./(2.*(ra + D/2.))
    h2 = D/2. -( ra + D/2. ) - r
    h3 = r - ( ra + D/2. ) - D/2.
    h4 = r + D/2. - ( ra + D/2. )
    h5 = ( ra + D/2. ) + D/2. + r
    
    h = h1*np.sqrt(h2*h3*h4*h5)

    theta = np.arcsin(h/r)
    
    S = 2.*pi*(r**2)*(-np.cos(theta) + 1)
    
    return S


def integral_of_surfaces(alpha, Lambda, r_min, D): # from mathematica
        
    F = (2.0*np.exp(-1.0*(D+r_min)/Lambda)*pi*alpha*Lambda*(D + 2.*(Lambda + r_min))*(D + D*np.exp(1.*D/Lambda) + 2.*Lambda - 2.*Lambda*np.exp(1.*D/Lambda)))/((D + 2.*r_min)**2)
        
    return -1.0*F



def final_integralx(alpha, Lambda, D, drop_rad, drop_len, center_center_distance, drop_x0):
    
    R = center_center_distance
    
    L = drop_len
    
    def fx(y, phi, r):
        
        r_min = -D/2. + np.sqrt(  R**2 + y**2 + r**2 - 2.*R*r*np.cos(phi)  )
        
        beta = 1.0*np.arcsin(  np.sqrt( (r*np.sin(phi))**2 + y**2 )/(r_min + D/2.) )
        
        ii = 1.0*r*integral_of_surfaces(alpha, Lambda, r_min, D)
        
        return np.cos(beta)*ii
    
    intex = integrate.tplquad(fx, 0., drop_rad, lambda r: 0., lambda r: 2.*pi, lambda r, phi: drop_x0, lambda r, phi: drop_x0 + L, epsrel = 1e-8, epsabs = 0)
    
    return intex


def final_integraly(alpha, Lambda, D, drop_rad, drop_len, center_center_distance, drop_x0):
    
    R = center_center_distance
    
    L = drop_len

    def fy(y, phi, r):
        
        r_min = -D/2. + np.sqrt(  R**2 + y**2 + r**2 - 2.*R*r*np.cos(phi)  )
        
        beta = 1.0*np.arcsin(  np.sqrt( (r*np.sin(phi))**2 + y**2 )/(r_min + D/2.) )
        
        ii = 1.0*r*integral_of_surfaces(alpha, Lambda, r_min, D)
        
        return 1.*(y/(r_min + D/2.))*ii

    intey = integrate.tplquad(fy, 0., drop_rad, lambda r: 0., lambda r: 2.*pi, lambda r, phi: drop_x0, lambda r, phi: drop_x0 + L, epsrel = 1e-8, epsabs = 0)
    
    return intey



def Forcex(rho_drop, rho_sphe, alpha, Lambda, D, drop_rad, drop_len, center_center_distance, drop_x0): # return Fx and Fy
    
    f = (1.0*G*rho_drop*rho_sphe)*np.array(final_integralx(alpha, Lambda, D, drop_rad, drop_len, center_center_distance, drop_x0))

    return f

def Forcey(rho_drop, rho_sphe, alpha, Lambda, D, drop_rad, drop_len, center_center_distance, drop_x0): # return Fx and Fy
    
    f = (1.0*G*rho_drop*rho_sphe)*np.array(final_integraly(alpha, Lambda, D, drop_rad, drop_len, center_center_distance, drop_x0))

    return f


# Sph_diam = 1.5e-5

# Drop_rad = 2.5e-5

# Drop_len = 1.0e-5

# dist = 1.0e-1

# center_center_distance = Drop_rad + Sph_diam/2. + 5.0e-6 + dist

# rho_drop = 6400. - 1000.

# rho_sphe = 1800.

# mass_sph = (4./3.)*pi*((Sph_diam/2.)**3)*rho_sphe

# mass_drop = pi*(Drop_rad**2)*Drop_len*rho_drop

   
# print ("result in ng")
# a = Forcex(rho_drop, rho_sphe, 1.0, 1.0e0, Sph_diam, Drop_rad, Drop_len, center_center_distance, -Drop_len/2.)[0]/((9.8e-9)*mass_sph)
# print (a)


# #pure newton acc infinity cylinder
# print "pure newton"
# gg = ((-G*mass_drop/(center_center_distance**2))/(9.8e-9))
# print gg

# print "ratio", 1.*a/gg
