import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
from scipy.special import wofz
import numpy as np
import glob
import scipy.optimize as opt
from scipy.stats import levy_stable
import matplotlib.cm as cm

pi = np.pi
folder_save = r"C:\data\20191122\10um\2\temp_x9"
folder_temp = r"C:\data\20191122\10um\2\temp_x9"
folder_nosphere = r"C:\data\20191122\10um\2\nosphere"

trans_out = (26./30.)**2
trans_in = (41./70.)**2 # this is the square of the power at the photodiode with sphere in and out.

plotHP = False
plotLP = False
plotnosphere = False

freq_fit_min = 40.
freq_fit_max = 90.


Diameter = 10.0e-6
rho = 1800

kb = 1.380e-23

threshould = 1e-9

fmax = 75.
gmax = 70.

def ffit(x, a, b):
    y = x*a + b
    m = fmax
    return min(y, m)

def gfit(x, a, b):
    y = x*a + b
    m = gmax
    return min(y, m)

freq_coef = [14.62066104, 63.1201561] # this is the estimation for the res freq as the feedback increases.
gamma_coef = [69.7547278, 0.09586861]

LT = 1 # plot line thickness

def get_folder_list(folder_temp):
    n = os.listdir(folder_temp)
    A = []
    for i in n:
        a = folder_temp + "\\" + i 
        A.append(a)
    return A

folder_list = get_folder_list(folder_temp)

def psd3(f,A,f0,gam,sig):

    omega = 2*np.pi*f
    omega_0 = 2*np.pi*f0

    z = ((omega**2 - omega_0**2) + 1j * omega*gam)/(np.sqrt(2)*sig)

    V = np.abs(A*np.real( wofz(z) )/sig)

    return np.sqrt(V)

def harmonic(f, f0, A, gamma):
    w0 = 2.*np.pi*np.abs(f0)
    w = 2.*np.pi*f
    gamma = 2.0*np.pi*gamma

    a1 = 1.*np.abs(A)
    a3 = 1.*(w0**2 - w**2)**2 + (w*gamma)**2

    s = 1.*a1/a3

    return s


def mass(Diameter, rho):
    m = (4/3.)*(np.pi)*((Diameter/2)**3)*rho
    return m

mass = mass(Diameter, rho)

list_of_plots = [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
#list_of_plots = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

import matplotlib.cm as cm
# Dark2 =  cm.get_cmap('viridis', len(np.logical_and(np.array(list_of_plots) > 0.9, np.array(list_of_plots) < 1.1)))
# colors = Dark2.colors
colors = ['#1f78b4', '#e66101', '#33a02c', '#984ea3', '#F27781', '#18298C', '#04BF8A', '#F2CF1D', '#F29F05', '#7155D9', '#8D07F6', '#9E91F2', '#F29B9B', '#F25764', '#6FB7BF', '#B6ECF2', '#5D1314', '#B3640F']

def selected_plot(folder_list, folder_nosphere, list_of_plots):
    
    name_load_nosphere = str(folder_nosphere) + "\info_outloop.npy"
    data_no_sphereout = np.load(name_load_nosphere)

    freqLP = data_no_sphereout[1]
    xpsd_nosphereout = np.sqrt(data_no_sphereout[3]*trans_out)

    index0 = np.where( freqLP >= freq_fit_min )[0][0]
    index1 = np.where( freqLP >= freq_fit_max )[0][0]
    fit_points1 = np.logical_and(freqLP > freq_fit_min, freqLP < 40.2)
    fit_points2 = np.logical_and(freqLP > 40.75, freqLP < 43.1)
    fit_points3 = np.logical_and(freqLP > 43.3, freqLP < 43.7)
    fit_points4 = np.logical_and(freqLP > 43.85, freqLP < 47.4)       
    fit_points5 = np.logical_and(freqLP > 47.8, freqLP < 48.4)
    fit_points6 = np.logical_and(freqLP > 49.35, freqLP < 49.9)
    fit_points7 = np.logical_and(freqLP > 50.83, freqLP < 54.6)        
    fit_points8 = np.logical_and(freqLP > 54.9, freqLP < 57.1)
    fit_points9 = np.logical_and(freqLP > 57.5, freqLP < 57.6)
    fit_points10 = np.logical_and(freqLP > 57.9, freqLP < 58.08)
    fit_points11 = np.logical_and(freqLP > 58.7, freqLP < 58.9)
    fit_points12 = np.logical_and(freqLP > 59.75, freqLP < 59.75)
    fit_points13 = np.logical_and(freqLP > 60.43, freqLP < 63.4)
    fit_points14 = np.logical_and(freqLP > 63.6, freqLP < 65.5)
    
    fit_points15 = np.logical_and(freqLP > 65.7, freqLP < 66.7)
    fit_points16 = np.logical_and(freqLP > 67.5, freqLP < 72.15)
    fit_points17 = np.logical_and(freqLP > 73.55, freqLP < 79.3)

    fit_points18 = np.logical_and(freqLP > 79.5, freqLP < 79.8)
    fit_points19 = np.logical_and(freqLP > 80.2, freqLP < 80.8)
    fit_points20 = np.logical_and(freqLP > 81.4, freqLP < 83.8)
    fit_points21 = np.logical_and(freqLP > 84.3, freqLP < 86.)
    fit_points22 = np.logical_and(freqLP > 86.6, freqLP < 88.8)
    fit_points23 = np.logical_and(freqLP > 89.5, freqLP < 89.7)

    fit_points24 = np.logical_and(freqLP > 89.91, freqLP < freq_fit_max)
        
    
    fit_points = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8 + fit_points9 + fit_points10 + fit_points11 + fit_points12 + fit_points13 + fit_points14 + fit_points15 + fit_points16 + fit_points17 + fit_points18 + fit_points19 + fit_points20 + fit_points21 + fit_points22 + fit_points23 + fit_points24
    not_fit_points =  np.logical_not(fit_points)
    
    plt.figure(figsize=(5.5,5))
    plt.rcParams.update({'font.size': 14})
    plt.subplot(2, 1, 1)
    plt.xlim(freq_fit_min, freq_fit_max)

    temp_out = []
    Dout = []
    gammaout = []
    fresout = []

    
    poptN, pcovN = opt.curve_fit(lambda f, T:  harmonic(f, fmax, T, gmax), freqLP[fit_points], (xpsd_nosphereout**2)[fit_points]/(2*pi))
    poptN = [fmax, poptN[0], gmax]
    toutN = ((fmax*2.*np.pi)**2)*mass*( np.sum(2.*np.pi*harmonic(freqLP, *poptN))*(freqLP[1] - freqLP[0])/(pi*kb) )/1e-6

    savenosphereout = harmonic(freqLP, *poptN)
    
    for i in range(len(folder_list)):
        if True:
            
            name_loadout = str(folder_list[i]) + "\info_outloop.npy"
            dataout = np.load(name_loadout)
            freqLP = dataout[1]
            xpsdout = np.sqrt(dataout[3])
            Dxout = dataout[6]
            Dout.append(Dxout)

            f_ex = ffit(np.abs(Dxout), *freq_coef)
            g_ex = gfit(np.abs(Dxout), *gamma_coef)

            if max(xpsdout[fit_points]) > threshould:
                poptQ, pcovQ = opt.curve_fit(psd3, freqLP[fit_points], np.sqrt(dataout[3][fit_points])/(2*pi), p0 = [2.42100889e-12, 6.32709456e+01 ,1.79414730e-02, 6.99109858e+02])
                tout = ((poptQ[1]*2.*np.pi)**2)*mass*( 2.*pi*np.sum((psd3(freqLP, *poptQ))**2)*(freqLP[1] - freqLP[0])/(pi*kb) )/1e-6
                g = poptQ[2]
                f = poptQ[1]
                if list_of_plots[i] == 1:
                    plt.semilogy(freqLP[index0:index1], 2.*np.pi*psd3(freqLP[index0:index1], *poptQ ), color = colors[i], linewidth=LT)
            else:

                try:
                    popt, pcov = opt.curve_fit(harmonic, freqLP[fit_points], dataout[3][fit_points]/(2*pi), p0 = [64, 0.7e-12, 3])
                    g = popt[2]
                    f = popt[0]
                    if list_of_plots[i] == 1:
                        plt.semilogy(freqLP[index0:index1], np.sqrt( 2.*np.pi*harmonic(freqLP[index0:index1], *popt)) , color = colors[i], linewidth=LT)
                    tout = ((popt[0]*2.*np.pi)**2)*mass*( np.sum(2.*np.pi*harmonic(freqLP, *popt))*(freqLP[1] - freqLP[0])/(pi*kb) )/1e-6

                    if np.abs(f) - f_ex > 0.05*f_ex or np.abs(g) - g_ex > 0.05*g_ex:
                        print "raising exception due to a bad fit OUT"
                        raise Exception("raising exception due to a bad fit")
                    
                except:
                    popt, pcov = opt.curve_fit(lambda f, T:  harmonic(f, f_ex, T, g_ex), freqLP[fit_points], dataout[3][fit_points]/(2*pi))
                    
                    popt = [f_ex, popt[0], g_ex]
                    g = popt[2]
                    f = popt[0]
                    tout =  ((f_ex*2.*np.pi)**2)*mass*( np.sum(2.*np.pi*harmonic(freqLP, *popt))*(freqLP[1] - freqLP[0])/(pi*kb) )/1e-6
                    if list_of_plots[i] == 1:
                        plt.semilogy(freqLP[index0:index1], np.sqrt( 2.*np.pi*harmonic(freqLP[index0:index1], *popt)) , color = colors[i], linewidth=LT)
            fresout.append(f)
            gammaout.append(g)
            temp_out.append(tout)
            if list_of_plots[i] == 1:
                plt.scatter(freqLP[fit_points], xpsdout[fit_points], marker = ".", color = colors[i], s = 4)
                plt.scatter(freqLP[not_fit_points], xpsdout[not_fit_points], marker = ".", alpha = 0.2, color = colors[i],s = 4)
    plt.semilogy(freqLP[index0:index1], xpsd_nosphereout[index0:index1], label = "Noise", color = "#add8e6")
    plt.semilogy(freqLP[index0:index1], np.sqrt( 2.*np.pi*harmonic(freqLP[index0:index1], *poptN)) , linewidth=LT, color = "#add8e6")
    plt.legend()
    plt.ylabel("$\sqrt{S_{xx}^{out}}$ [m/$\sqrt{Hz}$]")
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) 
    #plt.xlabel("Frequency [Hz]")
    # plt.tight_layout(pad = 0)
    # plt.text(41, 1e-8, "Outloop sensor")
    plt.grid()
    plt.ylim(3e-12, 1e-7)
    name = "outloop_" + ".pdf"
    name = os.path.join(folder_save, name)
        # plt.savefig(name)


#################inloop
    name_load_nosphere = str(folder_nosphere) + "\info_inloop.npy"
    data_no_sphere = np.load(name_load_nosphere)
    not_fit_points =  np.logical_not(fit_points)
    freqLP = data_no_sphere[1]
    xpsd_nosphere = np.sqrt(data_no_sphere[3]*trans_in)
    plt.subplot(2, 1, 2)
    plt.xlim(freq_fit_min, freq_fit_max)

    temp_in = []

    poptN, pcovN = opt.curve_fit(lambda f, T:  harmonic(f, fmax, T, gmax), freqLP[fit_points], (xpsd_nosphere**2)[fit_points]/(2*pi))
    poptN = [fmax, poptN[0], gmax]
    tinN = ((fmax*2.*np.pi)**2)*mass*( np.sum(2.*np.pi*harmonic(freqLP, *poptN))*(freqLP[1] - freqLP[0])/(pi*kb) )/1e-6

    savenospherein = harmonic(freqLP, *poptN)

    for i in range(len(folder_list)):
       if True:
            name_load = str(folder_list[i]) + "\info_inloop.npy"
            data = np.load(name_load)
            f = data[1]
            xpsd = np.sqrt(data[3])
            Dx = data[6]

            f_ex = ffit(np.abs(Dxout), *freq_coef)
            g_ex = gfit(np.abs(Dxout), *gamma_coef)

            if max(xpsd[fit_points]) > threshould:
                poptQ, pcovQ = opt.curve_fit(psd3, freqLP[fit_points], np.sqrt(data[3][fit_points])/(2*pi), p0 = [2.42100889e-12, 6.32709456e+01 ,1.79414730e-02, 6.99109858e+02])
                tin = ((poptQ[1]*2.*np.pi)**2)*mass*( np.sum(2.*np.pi*(psd3(freqLP, *poptQ))**2)*(freqLP[1] - freqLP[0])/(pi*kb) )/1e-6
                if list_of_plots[i] == 1:
                    plt.semilogy(freqLP[index0:index1], 2.*np.pi*psd3(freqLP[index0:index1], *poptQ ), color = colors[i], linewidth=LT)
            else:

                try:
                    popt, pcov = opt.curve_fit(harmonic, freqLP[fit_points], data[3][fit_points]/(2*pi), p0 = [64, 0.7e-12, 3])
                    if list_of_plots[i] == 1:
                        plt.semilogy(freqLP[index0:index1], np.sqrt( 2.*np.pi*harmonic(freqLP[index0:index1], *popt)) , color = colors[i], linewidth=LT)
                    tin = ((popt[0]*2.*np.pi)**2)*mass*( np.sum(2.*np.pi*harmonic(freqLP, *popt))*(freqLP[1] - freqLP[0])/(pi*kb) )/1e-6

                    if np.abs(popt[0]) - f_ex > 0.05*f_ex or np.abs(popt[2]) - g_ex > 0.05*g_ex:
                        print "raising exception due to a bad fit OUT"
                        raise Exception("raising exception due to a bad fit")
                except:
                    popt, pcov = opt.curve_fit(lambda f, T:  harmonic(f, f_ex, T, g_ex), freqLP[fit_points], data[3][fit_points]/(2*pi))
                    popt = [f_ex, popt[0], g_ex]
                    tin =  ((f_ex*2.*np.pi)**2)*mass*( np.sum(2.*np.pi*harmonic(freqLP, *popt))*(freqLP[1] - freqLP[0])/(pi*kb) )/1e-6

                    if list_of_plots[i] == 1:
                        plt.semilogy(freqLP[index0:index1], np.sqrt( 2.*np.pi*harmonic(freqLP[index0:index1], *popt)) , color = colors[i], linewidth=LT)

            temp_in.append(tin)
            if list_of_plots[i] == 1:
                plt.scatter(freqLP[fit_points], xpsd[fit_points], marker = ".", color = colors[i], s = 4)
                plt.scatter(freqLP[not_fit_points], xpsd[not_fit_points], marker = ".", alpha = 0.2, color = colors[i],s = 4)
    plt.semilogy(freqLP[index0:index1], xpsd_nosphere[index0:index1], color = "#add8e6")
    plt.semilogy(freqLP[index0:index1], np.sqrt( 2.*np.pi*harmonic(freqLP[index0:index1], *poptN)) , linewidth=LT, color = "#add8e6")
    plt.ylabel("$\sqrt{S_{xx}^{in}}$ [m/$\sqrt{Hz}$]")
    plt.xlabel("Frequency [Hz]")
    plt.tight_layout(pad = 0)
    # plt.text(41, 1e-8, "Inloop sensor")
    plt.grid()
    plt.ylim(3e-12, 1e-7)
    name = "inloop_" + ".pdf"
    name = os.path.join(folder_save, name)

    return [temp_out, temp_in, Dout, gammaout, fresout, toutN, tinN, savenosphereout, savenospherein, freqLP, fit_points]

tout, tin, D, g, f, toutN, tinN, psdnosphereout, psdnospherein, freq, fit_points = selected_plot(folder_list, folder_nosphere, list_of_plots)


space = np.linspace(0, 1.8, 100)
tout = np.array(tout)
tin = np.array(tin)
D = np.abs(np.array(D))

touterr = 0.4*tout
tinerr = 0.4*tin

plt.figure(figsize=(5,3))
plt.rcParams.update({'font.size': 14})
plt.errorbar(100*D, tout, yerr = touterr, fmt = "o", label = "Outloop")
plt.errorbar(100*D, tin, yerr = tinerr, fmt = "o", label = "Inloop")
# plt.hlines(toutN, np.min(D)-2e-3, np.max(D)+2e-3)
# plt.hlines(tinN, np.min(D)-2e-3, np.max(D)+2e-3)
plt.xlabel("Derivative Gain [Arb. Units]")
plt.ylabel("Temperature [$\mu$K]")
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid()
plt.tight_layout(pad = 0)

D = np.abs(np.array(D))
g = np.abs(np.array(g))

fit_points = np.logical_and(np.abs(D) > 0, np.abs(D) < 0.35)

pg = np.polyfit(D[fit_points], g[fit_points], 1)

print "k =", pg


plt.figure()
plt.plot(D, g, ".")
plt.plot(D[fit_points], g[fit_points], "x")
plt.plot(space, space*pg[0] + pg[1])

# D = np.abs(np.array(D))
f = np.abs(np.array(f))
fit_points = np.logical_and(np.abs(D) > 0, np.abs(D) < 0.35)

pf = np.polyfit(D[fit_points], f[fit_points], 1)
space = np.linspace(0, 1.8, 100)
print "H =", pf

plt.figure()
plt.plot(np.abs(D), np.abs(f), "x")
plt.plot(space, space*pf[0] + pf[1])




############################## fit model:

def Sc(freq, fres, L, Dg, gain, Gamma, nosphere_psd_out, TIME):
    
    freq = 2.*pi*freq
    fres = 2.*pi*fres
    Gamma = 2.*pi*Gamma
    nosphere_psd_out = 1.*nosphere_psd_out/(2.*pi)
    nosphere_spectrum = nosphere_psd_out*((freq[1] - freq[0])*2.*pi) # not density anymore

    H = -1j*(2.*pi)*freq

    S = []

    for i in range(len(Dg)):
            Dg = 1.*Dg*gain
    
            aux1 = 1.*L**2 + 2.*L*(fres[i]**2)*Dg[i]*np.real(H)*nosphere_spectrum + (fres[i]**4)*(Dg[i]**2)*(np.abs(H)**2)*(nosphere_spectrum**2)
            
            aux2 = 1.*(freq**2 - fres[i]**2)**2 - 2.*(freq**2 - fres[i]**2)*Dg[i]*np.real(H) - 2.*Gamma[i]*freq*(fres[i]**2)*Dg[i]*np.imag(H) + 1.*(fres[i]**4)*(Dg[i]**2)*(np.abs(H))**2 + 1.*(Gamma[i]*freq)**2
    
            s = (1./TIME)*(aux1/aux2)
            s = np.sum(s)*((freq[1] - freq[0])*2.*pi)
            S.append(s)

    return np.array(S)





def Tc(freq, fres, L, Dg, Gamma, TIME, nosphere_psd_out, mass, kb, gain):
    S = Sc(freq, fres, L, Dg, gain, Gamma, nosphere_psd_out, TIME)

    Tc = ((fres*2.*np.pi)**2)*mass*S/(kb*pi)

    Tc = Tc/1e6

    return Tc

name = "squasing_info.npy"
name = os.path.join(folder_save, name)
np.save(name, [freq, f, 2., D, g, 2**19/1e4, psdnosphereout, mass, kb, 3., tout, toutN, tin, tinN, psdnospherein])
    
print Tc(freq, f, 2., D, g, 2**19/1e4, psdnosphereout, mass, kb, 3.)

def test(D, L, gain):
    return Tc(freq, f, L, D, g, 2**19/1e4, psdnosphereout, mass, kb, gain)

popt, pcov = opt.curve_fit(test, D, tout, p0 = [2 ,0.000001])


plt.figure(figsize=(5,3))
plt.rcParams.update({'font.size': 14})
plt.errorbar(100*D, tout, yerr = touterr, fmt = "o", label = "Outloop")
plt.errorbar(100*D, tin, yerr = tinerr, fmt = "o", label = "Inloop")
plt.plot(100.*D, test(D, *popt), "k--")
plt.xlabel("Derivative Gain [Arb. Units]")
plt.ylabel("Temperature [$\mu$K]")
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid()
plt.tight_layout(pad = 0)


plt.show()
