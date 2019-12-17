import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import glob
import scipy.optimize as opt

# inside the npy file: [freq, f, 2., D, g, 2**19/1e4, psdnosphereout, mass, kb, 3., tout, toutN, tin, tinN, psdnospherein]

# freq is the array of all frequencies
# f is useless (resonant freq used on the fit of the psd )
# D is the derivative gain used on the measuremets
# g is useless (gamma used for the fit of the psd)
# 2**19/1e4 is the measurement time
# psdnosphereout is the psd m2/Hz fir the noise without sphere at the outloop sensor
# psdnospherein is the psd m2/Hz fir the noise without sphere at the inloop sensor
# tout is the temperature of each measurement for the outloop sensor
# tin is the temperature of each measurement for the inloop sensor
# toutN and tinN are the noise equivalent temperature for each sensor.
# kb is boltzmann constant

pi = np.pi

folder_load = r"C:\data\20191122\10um\2\temp_x9"
name = "squasing_info.npy"
name = os.path.join(folder_load, name)
info = np.load(name)

freq = info[0]
D = info[3]
g = info[4]
psdnosphereout = np.sqrt(info[6])
mass = info[7]
kb = info[8]
tout = info[10]
toutN = info[11]
tin = info[12]
tinN = info[13]
psdnospherein = np.sqrt(info[14])

fmax = 75.
def ffit(x, a, b): # this is the resonance frequency as function of the feedback
    y = x*a + b
    m = fmax
    return min(y, m)

freq_coef = [14.62066104, 63.1201561] # this is the estimation for the res freq as the feedback increases.

### notice that below I am setting Gamma = 0 as in this equation this is the damping due to air.
### The parameter L is what heats the sphere.
### Time is the measurement time
### gain is the number that multiplies the feedback Dg

### Sc calculates the temp for the outloop
### Ss calculates the temp for the inloop
def Sc(freq, L, Dg, Gamma, nosphere_psd_out, TIME, gain, mass):
    
    freq = 2.*pi*freq
    nosphere_psd_out = 1.*nosphere_psd_out**2
    nosphere_spectrum = nosphere_psd_out*((freq[1] - freq[0])) # not density anymore
    nosphere_spectrum = np.sqrt(nosphere_spectrum)
    gamma = 1e-6 # this is zero...
    
    H = -1j*freq

    T = []

    for i in range(len(Dg)):
        fres = ffit(Dg[i], 14.62066104, 63.1201561)*2.*np.pi
        dg = 1.*Dg[i]*gain
        
        aux1 = 1.*L**2 + 2.*L*(fres**2)*dg*np.real(H)*nosphere_spectrum + (fres**4)*(dg**2)*(np.abs(H)**2)*(nosphere_spectrum**2)
            
        aux2 = 1.*(freq**2 - fres**2)**2 - 2.*(freq**2 - fres**2)*dg*np.real(H) + 1.*(fres**4)*(dg**2)*(np.abs(H))**2 + (gamma*freq)**2 - 2.*gamma*freq*(fres**2)*dg*np.imag(H)
    
        s = (1./TIME)*(aux1/aux2)
        s = np.sum(s)*((freq[1] - freq[0]))
        t1 = s*((fres)**2)*mass/(kb*pi)

        t = t1/1e-6
        
        T.append(t)

    return np.array(T)


def Ss(freq, L, Dg, Gamma, nosphere_psd_in, TIME, gain, mass):
    
    freq = 2.*pi*freq
    nosphere_psd_in = 1.*nosphere_psd_in**2
    nosphere_spectrum = nosphere_psd_in*((freq[1] - freq[0])) # not density anymore
    nosphere_spectrum = np.sqrt(nosphere_spectrum)
    gamma = 1e-6
    
    H = -1j*freq

    T = []

    for i in range(len(Dg)):
        fres = ffit(Dg[i], 14.62066104, 63.1201561)*2.*np.pi
        dg = 1.*Dg[i]*gain
        
        aux1 = 1.*L**2 + 2.*L*((fres**2 - freq**2))*nosphere_spectrum + (((fres**2 - freq**2)**2) + (gamma*freq)**2)*(nosphere_spectrum**2)
            
        aux2 = 1.*(freq**2 - fres**2)**2 - 2.*(freq**2 - fres**2)*dg*np.real(H) + 1.*(fres**4)*(dg**2)*(np.abs(H))**2 + (gamma*freq)**2 - 2.*gamma*freq*(fres**2)*dg*np.imag(H)
    
        s = (1./TIME)*(aux1/aux2)
        s = np.sum(s)*((freq[1] - freq[0]))
        t1 = s*((fres)**2)*mass/(kb*pi)

        t = t1/1e-6
        
        T.append(t)

    return np.array(T)



ind = np.argsort(D) # sorting D, not really necessary
D = D[ind]
tout = tout[ind]
tin = tin[ind]

# D = D[2:-1]
# tout = tout[2:-1]
# tin = tin[2:-1]


# below is errorbar due to mass
touterr = 0.4*tout
tinerr = 0.4*tin

popt, pcov = opt.curve_fit(lambda D, L, gain: Sc(freq, L, D, g, psdnosphereout, 2**19/1e4, gain, mass), D, tout, sigma = touterr)
print popt


popts, pcovs = opt.curve_fit(lambda D, L, gain: Ss(freq, L, D, g, psdnospherein, 2**19/1e4, gain, mass), D, tin, sigma = tinerr)
print popts

def test(D, L, gain): # not the best procedure but works
    return Sc(freq, L, D, g, psdnosphereout, 2**19/1e4, gain, mass)

def test_s(D, L, gain):
    return Ss(freq, L, D, g, psdnospherein, 2**19/1e4, gain, mass)

Dlist = 3.*np.logspace(-3, 0, 100)

plt.figure(figsize=(4,3))
plt.rcParams.update({'font.size': 14})
plt.errorbar(D, tout, yerr = touterr, fmt = "o", label = "Outloop", color = "C0")
plt.errorbar(D, tin, yerr = tinerr, fmt = "o", label = "Inloop",  color = "C1")
plt.plot(Dlist, test(Dlist, *popt), "-", color = "C0", alpha = 0.5, linewidth = 2)
plt.plot(Dlist, test_s(Dlist, *popts), "-", color = "C1", alpha = 0.5, linewidth = 2)
plt.xlabel("Derivative Gain [Arb. Units]")
plt.ylabel("Temperature [$\mu$K]")
plt.yscale('log')
plt.xscale('log')
plt.xlim(0.003,2)
plt.legend()
plt.grid()
plt.tight_layout(pad = 0)


plt.show()
