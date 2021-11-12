import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
from scipy.special import wofz
import numpy as np
import glob
import scipy.optimize as opt
from scipy.stats import levy_stable

pi = np.pi
# freqhp, freq_meas, xpsd2_m_hp, xpsd2_m_meas, xpsd2_v_hp, xpsd2_meas_v, xDg

folder_save = r"C:\data\20191107\22um\results\9"
folder_temp = r"C:\data\20191107\22um\9\temp_x2"
folder_nosphere = r"C:\data\20191107\22um\9\nosphere"

trans_out = (42./52.)**2
trans_in = (41./70.)**2 # this is the square of the power at the photodiode with sphere in and out.

folder_save = r"C:\data\20191119\10um\4\temp_x2"
folder_temp = r"C:\data\20191119\10um\4\temp_x2"
folder_nosphere = r"C:\data\20191119\10um\4\nosphere"
# fit_points1 = np.logical_and(freqLP > freq_fit_min, freqLP < 64.9)
# fit_points2 = np.logical_and(freqLP > 65.8, freqLP < 71.2)
# fit_points3 = np.logical_and(freqLP > 72.1, freqLP < 80.8)
# fit_points4 = np.logical_and(freqLP > 81.5, freqLP < 83.3)
# fit_points5 = np.logical_and(freqLP > 84.2, freqLP < freq_fit_max)

trans_out = (25./28.)**2
trans_in = (41./70.)**2 # this is the square of the power at the photodiode with sphere in and out.


folder_save = r"C:\data\20191122\10um\2\temp_x9"
folder_temp = r"C:\data\20191122\10um\2\temp_x9"
folder_nosphere = r"C:\data\20191122\10um\2\nosphere"

trans_out = (26./30.)**2
trans_in = (41./70.)**2 # this is the square of the power at the photodiode with sphere in and out.

plotHP = False
plotLP = False
plotnosphere = False

fit_true_psd_false = False # how to calculate the energy


freq_plot_min = 60.
freq_plot_max = 90.

# freq_fit_min = 60.
# freq_fit_max = 110.

freq_fit_min =61.
freq_fit_max = 90.

freq_fit_min =40.
freq_fit_max = 80.

freq_fit_min = 40.
freq_fit_max = 90.

Diameter = 22.6e-6 # meters
Diameter = 10.0e-6
rho = 1800

kb = 1.380e-23

def get_folder_list(folder_temp):
    n = os.listdir(folder_temp)
    A = []
    for i in n:
        a = folder_temp + "\\" + i 
        A.append(a)
    return A

folder_list = get_folder_list(folder_temp)

def mass(Diameter, rho):
    m = (4/3.)*(np.pi)*((Diameter/2)**3)*rho
    return m

mass = mass(Diameter, rho)

def line(f, a, b):
    return f*a + b

def psd(f, A, f0, gamma):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    gamma1 = 2.0*np.pi*gamma
    s1 = 2.*A*(gamma1*(w0**2))
    s2 = 1.*(w0**2)*((w0**2 - w**2)**2 + (gamma1*w)**2)
    s = np.sqrt(s1/s2)
    return A*s



def psd2(f, A, f0, gamma):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    gamma1 = 2.0*np.pi*gamma
    s1 = 2.*A*(gamma1*(w0**2))
    s2 = 1.*(w0**2)*((w0**2 - w**2)**2 + (gamma1*w)**2)
    s = np.sqrt(s1/s2)
    return (A*s)**2


def psd3(f,A,f0,gam,sig):

    omega = 2*np.pi*f
    omega_0 = 2*np.pi*f0

    z = ((omega**2 - omega_0**2) + 1j * omega*gam)/(np.sqrt(2)*sig)

    V = np.abs(A*np.real( wofz(z) )/sig)

    return np.sqrt(V)


def harmonic_HP(f, f0, T, gamma):
    w0 = 2.*np.pi*np.abs(f0)
    w = 2.*np.pi*f
    gamma = 2.0*np.pi*gamma

    a1 = 2.*kb*T/mass
    a2 = 1.*gamma
    a3 = 1.*(w0**2 - w**2)**2 + (w*gamma)**2

    s = 1.*a1*a2/a3

    return s

def harmonic(f, f0, A, gamma):
    w0 = 2.*np.pi*np.abs(f0)
    w = 2.*np.pi*f
    gamma = 2.0*np.pi*gamma

    a1 = 1.*np.abs(A)
    a3 = 1.*(w0**2 - w**2)**2 + (w*gamma)**2

    s = 1.*a1/a3

    return s


def temperatude_folder_HP(folder_temp, loop):
    if loop == "out":
        name_load = str(folder_temp) + "\info_outloop.npy"
    else:
        name_load = str(folder_temp) + "\info_inloop.npy"
    data = np.load(name_load)
    
    freqHP = data[0]
    xpsd2_m_HP = data[2]
    
    index0 = np.where( freqHP >= freq_fit_min )[0][0]
    index1 = np.where( freqHP >= freq_fit_max )[0][0]
    
    fit_points1 = np.logical_and(freqHP > freq_fit_min, freqHP < 59.0)
    fit_points2 = np.logical_and(freqHP > 61.0, freqHP < freq_fit_max)
    fit_points = fit_points1 + fit_points2
    
    popt, pcov = opt.curve_fit(harmonic_HP, freqHP[fit_points], xpsd2_m_HP[fit_points]/(2.*pi), p0 = [60, 500, 500])
    
    if plotHP:
        plt.figure()
        plt.loglog(freqHP, xpsd2_m_HP)
        plt.loglog(freqHP, 2.*pi*harmonic(freqHP, *popt))
    
    temp = popt[1]

    # total_displacement2 = np.sum( 2.*pi*harmonic(freqHP, *popt))*(freqHP[1] - freqHP[0])
    # f0 = np.abs(popt[0])
    # spring_constant = mass*((2.*np.pi*f0)**2)
    # mean_energy_spring = 0.5*spring_constant*total_displacement2/pi # the factor pi comes from tongcang thesis
    # temp1 = 2.0*mean_energy_spring/kb

    print temp
    
    return [temp, popt[0]]




def temperatude_folder_nosphere(folder, freq, Diameter, loop): # for no sphere, the temp comes from the area of the data, not the fit
    if loop == "out":
        name_load = str(folder) + "\info_outloop.npy"
    else:
        name_load = str(folder) + "\info_inloop.npy"
    data = np.load(name_load)
    
    freqLP = data[1]
    xpsd2_m_LP = data[3]
    
    xDg = data[6]
    
    index0 = np.where( freqLP >= freq_fit_min )[0][0]
    index1 = np.where( freqLP >= freq_fit_max )[0][0]

    if Diameter == 22.8e-6:
        fit_points1 = np.logical_and(freqLP > freq_fit_min, freqLP < 56.5)
        fit_points2 = np.logical_and(freqLP > 60.6, freqLP < 64.8)
        fit_points3 = np.logical_and(freqLP > 65.5, freqLP < 72.3)
        fit_points4 = np.logical_and(freqLP > 73.0, freqLP < 80.0)
        fit_points5 = np.logical_and(freqLP > 81.5, freqLP < 83.3)
        fit_points6 = np.logical_and(freqLP > 84.2, freqLP < 99.)
        fit_points7 = np.logical_and(freqLP > 102, freqLP < 108.8)
        fit_points8 = np.logical_and(freqLP > 109.5, freqLP < freq_fit_max)
    
        fit_points = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8
    
    if Diameter == 10.0e-6:
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

    popt, pcov = opt.curve_fit(lambda f, T, gamma:  harmonic(f, 72., T, gamma), freqLP[fit_points], xpsd2_m_LP[fit_points]/(2*pi), p0 = [1e-3, 1000.])
    popt = [75., popt[0], popt[1]]
        
    if plotnosphere:
        plt.figure()
        plt.loglog(freqLP[index0:index1], xpsd2_m_LP[index0:index1])
        plt.loglog(freqLP[index0:index1], 2*np.pi*harmonic(freqLP[index0:index1], *popt))
    
    df = freqLP[1] - freqLP[0]
    total_displacement2 = np.sum( 2.*pi*harmonic(freqLP, *popt))*df
    
    spring_constant = mass*((2.*np.pi*popt[0])**2)
    
    mean_energy_spring = 0.5*spring_constant*total_displacement2/pi
    
    temp = 2.0*mean_energy_spring/kb
    
    return [temp, xDg, xpsd2_m_LP]




def temperatude_folder(folder, freq, fit_vs_psd, Diameter, loop, noise_psd2):
    if loop == "out":
        noise_psd2 = trans_out*noise_psd2
        name_load = str(folder) + "\info_outloop.npy"
    else:
        noise_psd2 = np.zeros(len(noise_psd2))
        name_load = str(folder) + "\info_inloop.npy"
    data = np.load(name_load)
    
    freqLP = data[1]
    xpsd2_m_LP = data[3]
    xpsd2_m_LP = xpsd2_m_LP - noise_psd2
    
    xDg = data[6]
    
    index0 = np.where( freqLP >= freq_fit_min )[0][0]
    index1 = np.where( freqLP >= freq_fit_max )[0][0]

    if Diameter == 22.8e-6:
        fit_points1 = np.logical_and(freqLP > freq_fit_min, freqLP < 56.5)
        fit_points2 = np.logical_and(freqLP > 60.6, freqLP < 64.8)
        fit_points3 = np.logical_and(freqLP > 65.5, freqLP < 72.3)
        fit_points4 = np.logical_and(freqLP > 73.0, freqLP < 80.0)
        fit_points5 = np.logical_and(freqLP > 81.5, freqLP < 83.3)
        fit_points6 = np.logical_and(freqLP > 84.2, freqLP < 99.)
        fit_points7 = np.logical_and(freqLP > 102, freqLP < 108.8)
        fit_points8 = np.logical_and(freqLP > 109.5, freqLP < freq_fit_max)
    
        fit_points = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8
    
    if Diameter == 10.0e-6:
        fit_points1 = np.logical_and(freqLP > freq_fit_min, freqLP < 40.2)
        fit_points2 = np.logical_and(freqLP > 40.75, freqLP < 43.1)
        fit_points3 = np.logical_and(freqLP > 43.3, freqLP < 43.7)
        fit_points4 = np.logical_and(freqLP > 43.85, freqLP < 47.4)       
        fit_points5 = np.logical_and(freqLP > 47.8, freqLP < 48.5)
        fit_points6 = np.logical_and(freqLP > 49.23, freqLP < 50.14)
        fit_points7 = np.logical_and(freqLP > 50.83, freqLP < 54.6)        
        fit_points8 = np.logical_and(freqLP > 54.9, freqLP < 57.1)
        fit_points9 = np.logical_and(freqLP > 57.5, freqLP < 57.6)
        fit_points10 = np.logical_and(freqLP > 57.9, freqLP < 58.15)
        fit_points11 = np.logical_and(freqLP > 58.7, freqLP < 59.28)
        fit_points12 = np.logical_and(freqLP > 59.6, freqLP < 59.75)
        fit_points13 = np.logical_and(freqLP > 60.43, freqLP < 63.4)
        fit_points14 = np.logical_and(freqLP > 63.6, freqLP < 66.7)
        fit_points15 = np.logical_and(freqLP > 67.5, freqLP < 72.15)
        fit_points16 = np.logical_and(freqLP > 73.55, freqLP < 79.3)
        fit_points17 = np.logical_and(freqLP > 79.5, freqLP < freq_fit_max)
        
    
        fit_points = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8 + fit_points9 + fit_points10 + fit_points11 + fit_points12 + fit_points13 + fit_points14 + fit_points15 + fit_points16 + fit_points17
#(f, f0, T, gamma)
    try:
        popt, pcov = opt.curve_fit(harmonic_HP, freqLP[fit_points], xpsd2_m_LP[fit_points]/(2*pi), p0 = [62., 1e-3, 10.])
        if popt[0] > 75.:
            popt, pcov = opt.curve_fit(lambda f, T, gamma:  harmonic_HP(f, 72., T, gamma), freqLP[fit_points], xpsd2_m_LP[fit_points]/(2*pi), p0 = [1e-3, 10.])
            popt = [75., popt[0], popt[1]]
        print popt
        f0 = popt[0]
    except:
        popt = [freq, 1e-4, 10]
        print "plot failed" + str(folder)
        f0 = freq


    if plotLP:
        plt.figure()
        label = str(xDg)+str(loop)+"_"+str(folder)
        plt.semilogy(freqLP[fit_points], xpsd2_m_LP[fit_points], label = label)
        # plt.plot(freqLP[np.logical_not(fit_points)], xpsd2_m_LP[np.logical_not((fit_points))], "rx", label = label)
        plt.semilogy(freqLP, 2.*pi*harmonic_HP(freqLP, *popt))
        # plt.plot(freqLP, noise_psd2)
        plt.xlim(freq_fit_min, freq_fit_max)
        plt.legend()

    df = freqLP[1] - freqLP[0]

    # total_displacement2 = np.sum(xpsd2_m_LP[fit_points])*df
    total_displacement2 = np.sum( 2.*pi*harmonic_HP(freqLP, *popt))*df
    fres = freq
    spring_constant = mass*((2.*np.pi*f0)**2)
    mean_energy_spring = 0.5*spring_constant*total_displacement2/pi # the factor pi comes from tongcang thesis
    temp = 2.0*mean_energy_spring/kb


    return [temp, xDg, popt[1], f0]



[HPout, fres] = temperatude_folder_HP(folder_list[0], "out")
[HPin, fres] = temperatude_folder_HP(folder_list[0], "in")
nosphere_out = temperatude_folder_nosphere(folder_nosphere, fres, Diameter, "out")
nosphere_in = temperatude_folder_nosphere(folder_nosphere, fres, Diameter, "in")

noise_in_psd2 = nosphere_in[2]
noise_out_psd2 = nosphere_out[2]


# nolaser_out = temperatude_folder_nosphere(folder_nolaser, fres, Diameter, "out")
# nolaser_in = temperatude_folder_nosphere(folder_nolaser, fres, Diameter, "in")

def linefit(x, a, b):
    return a*x + b

T = []
T_fit = []
D = []


T2 = []
T2_fit = []
D2 = []
f02 = []
for i in folder_list:
    a = temperatude_folder(i, fres, fit_true_psd_false, Diameter, "out", noise_out_psd2)
    T.append(a[0])
    T_fit.append(a[2])
    D.append(np.abs(a[1])+1e-6)

    a2 = temperatude_folder(i, fres, fit_true_psd_false, Diameter, "in", noise_in_psd2)
    T2.append(a2[0])
    T2_fit.append(a2[2])
    D2.append(np.abs(a2[1])+1e-6)
    f02.append(a[3])


# popt_D, pcov_D = opt.curve_fit(linefit, D2, f02)
# line = np.linspace(np.min(D2), np.max(D2), 1000)
plt.figure()
plt.plot(D2, f02, "r.")
# plt.plot(line, linefit(line, *popt_D))


    
plt.figure()
plt.loglog(D, 1.0e6*np.array(T), "ro", label = "outloop")
# plt.loglog(D, 1.0e6*np.array(T_fit), "rx", label = "outloop_fit")
plt.hlines(1e6*nosphere_out[0], np.min(D)-2e-3, np.max(D)+2e-3, color = "r", label = "no sphere outloop")
plt.hlines(1e6*HPout, np.min(D)-2e-3, np.max(D)+2e-3, label = "1mbar outloop")
plt.loglog(D, 1.0e6*np.array(T2), "bo", label = "inloop")
# plt.loglog(D, 1.0e6*np.array(T2_fit), "bx", label = "inloop_fit")
plt.hlines(1e6*nosphere_in[0], np.min(D)-2e-3, np.max(D)+2e-3, color = "b", label = "no sphere inloop")
plt.hlines(1e6*HPin, np.min(D2)-2e-3, np.max(D2)+2e-3, label = "1mbar inloop")
plt.ylabel("COM X temp [$\mu$ K]")
plt.xlabel("DG X")
plt.tight_layout(pad = 0)

# plt.hlines(1e6*nolaser_in[0], np.min(D)-2e-3, np.max(D)+2e-3, color = "b", label = "laser off", linestyle = "--")
# plt.hlines(1e6*nolaser_out[0], np.min(D)-2e-3, np.max(D)+2e-3, color = "r", label = "laser off", linestyle = "--")

plt.legend()
plt.grid()
name = "temp_" + ".pdf"
name = os.path.join(folder_save, name)
# plt.savefig(name)

plt.show()
