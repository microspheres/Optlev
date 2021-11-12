import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import glob
import scipy.optimize as opt
import return_xyzcool_yzcool as rt

# freqhp, freq_meas, xpsd2_m_hp, xpsd2_m_meas, xpsd2_v_hp, xpsd2_meas_v, xDg


# folder_list = [r"C:\data\20191022\10um\prechamber_LP\1\temp_x\1", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\2", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\3", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\4", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\5", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\6", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\7", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\8", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\9", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\10", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\11", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\13", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\14", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\15", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\16", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\17", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\18"]

# folder_nosphere = r"C:\data\20191022\10um\prechamber_LP\1\nosphere"

# folder_HP = r"C:\data\20191022\10um\prechamber_LP\1\2mbar"
# xyz = r"2mbar_xyzcool.h5"
# yz = r"2mbar_yzcool.h5"




# folder_list = [r"C:\data\paper2\22um\PreChamber_LP\3\temp_x\1", r"C:\data\paper2\22um\PreChamber_LP\3\temp_x\2", r"C:\data\paper2\22um\PreChamber_LP\3\temp_x\3", r"C:\data\paper2\22um\PreChamber_LP\3\temp_x\4", r"C:\data\paper2\22um\PreChamber_LP\3\temp_x\5", r"C:\data\paper2\22um\PreChamber_LP\3\temp_x\6", r"C:\data\paper2\22um\PreChamber_LP\3\temp_x\7", r"C:\data\paper2\22um\PreChamber_LP\3\temp_x\8", r"C:\data\paper2\22um\PreChamber_LP\3\temp_x\9", r"C:\data\paper2\22um\PreChamber_LP\3\temp_x\10", r"C:\data\paper2\22um\PreChamber_LP\3\temp_x\11", r"C:\data\paper2\22um\PreChamber_LP\3\temp_x\12", r"C:\data\paper2\22um\PreChamber_LP\3\temp_x\13", ]

# folder_nosphere = r"C:\data\paper2\22um\PreChamber_LP\2\nosphere"

# folder_HP = r"C:\data\paper2\22um\PreChamber_LP\3\1mbar"
# xyz = r"1mbar_xyzcool.h5"
# yz = r"1mbar_yzcool.h5"

folder_list = [r"C:\data\paper3\22um\PreChamber_ATM\2\temp_x\1", r"C:\data\paper3\22um\PreChamber_ATM\2\temp_x\2", r"C:\data\paper3\22um\PreChamber_ATM\2\temp_x\3", r"C:\data\paper3\22um\PreChamber_ATM\2\temp_x\4", r"C:\data\paper3\22um\PreChamber_ATM\2\temp_x\5", r"C:\data\paper3\22um\PreChamber_ATM\2\temp_x\6", r"C:\data\paper3\22um\PreChamber_ATM\2\temp_x\7", r"C:\data\paper3\22um\PreChamber_ATM\2\temp_x\8", r"C:\data\paper3\22um\PreChamber_ATM\2\temp_x\9", r"C:\data\paper3\22um\PreChamber_ATM\2\temp_x\10", r"C:\data\paper3\22um\PreChamber_ATM\2\temp_x\11", r"C:\data\paper3\22um\PreChamber_ATM\2\temp_x\12", r"C:\data\paper3\22um\PreChamber_ATM\2\temp_x\13"]

folder_nosphere = r"C:\data\paper3\22um\PreChamber_ATM\2\nosphere"

folder_HP = r"C:\data\paper2\22um\PreChamber_LP\3\1mbar"
xyz = r"1mbar_xyzcool.h5"
yz = r"1mbar_yzcool.h5"

folder_list = [r"C:\data\paper3\22um\PreChamber_LP\1\temp_x\1", r"C:\data\paper3\22um\PreChamber_LP\1\temp_x\2",r"C:\data\paper3\22um\PreChamber_LP\1\temp_x\3",r"C:\data\paper3\22um\PreChamber_LP\1\temp_x\4",r"C:\data\paper3\22um\PreChamber_LP\1\temp_x\5",r"C:\data\paper3\22um\PreChamber_LP\1\temp_x\6", r"C:\data\paper3\22um\PreChamber_LP\1\temp_x\7", r"C:\data\paper3\22um\PreChamber_LP\1\temp_x\8", r"C:\data\paper3\22um\PreChamber_LP\1\temp_x\9", r"C:\data\paper3\22um\PreChamber_LP\1\temp_x\10", r"C:\data\paper3\22um\PreChamber_LP\1\temp_x\11",]

folder_nosphere = r"C:\data\paper3\22um\PreChamber_LP\1\nosphere"

folder_HP = r"C:\data\paper3\22um\PreChamber_LP\1\1mbar"
xyz = r"1mbar_xyzcool.h5"
yz = r"1mbar_yzcool.h5"

gammaover2pi = 0.18

NFFT = 2**17

plotHP = True
plotLP = False
plotnosphere = False

fit_true_psd_false = False # how to calculate the energy

freq_plot_min = 30.
freq_plot_max = 130.

freq_fit_min = 50.
freq_fit_max = 120.

# freq_fit_min =45.
# freq_fit_max = 100.

Diameter = 22.8e-6 # meters
# Diameter = 10.0e-6
rho = 1800.

kb = 1.380e-23

cut_low = 730
cut_high = 1800

T0 = 1.7e7 #kelvin

def mass(Diameter, rho):
    m = (4/3.)*(np.pi)*((Diameter/2)**3)*rho
    return m

mass = mass(Diameter, rho)

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
    s1 = 2.*(gamma1*(w0**2))
    s2 = 1.*(w0**2)*((w0**2 - w**2)**2 + (gamma1*w)**2)
    s = np.sqrt(s1/s2)
    return A*(s)**2



def feedback_filter_squared(freq, cut_low, cut_high):
    a = 1.0/np.sqrt((1.0 + (freq/cut_low)**2))
    b = freq/np.sqrt((freq**2 + (cut_high)**2))
    H = a*b
    return H**2

def feedback_filter_real(freq, cut_low, cut_high):
    H = np.sqrt(feedback_filter_squared(freq, cut_low, cut_high))
    phase_low = -np.arctan(freq/cut_low)
    phase_high = np.pi/2. - np.arctan(freq/cut_high)
    return H*np.cos(phase_low + phase_high)

def feedback_filter_imaginary(freq, cut_low, cut_high):
    H = np.sqrt(feedback_filter_squared(freq, cut_low, cut_high))
    phase_low = -np.arctan(freq/cut_low)
    phase_high = np.pi/2. - np.arctan(freq/cut_high)
    return H*np.sin(phase_low + phase_high)

def psd2_HP_feedback(f, A, f0, gamma, cut_low, cut_high, xDg):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    gamma1 = 2.0*np.pi*gamma
    s1 = 2.*(gamma1*(w0**2))
    s2 = 1.*(w0**2)*((w0**2 - w**2)**2 + (gamma1*w)**2)
    s3 = -2.0*(w0**2 - w**2)*(w0**2)*xDg*feedback_filter_real(w, 2.*np.pi*cut_low, 2.*np.pi*cut_high)
    s4 = 2.0*gamma1*w*(w0**2)*xDg*feedback_filter_imaginary(w, 2.*np.pi*cut_low, 2.*np.pi*cut_high)
    s5 = (w0**4)*(xDg**2)*feedback_filter_squared(w, 2.*np.pi*cut_low, 2.*np.pi*cut_high)

    s2 = s2 + s3 + s4 + s5
    
    s = np.sqrt(s1/s2)
    return A*(s)**2


def max_between_array_ones(a):
    b = np.ones(len(a))
    c = []
    for i in range(len(a)):
        if a[i] > b[i]:
            c.append(a[i])
        else:
            c.append(b[i])
    c = np.array(c)
    return c
    
def extract_feedback_scale(folder_HP, xyz, yz, NFFT): # return the scale that multiplies the feedback X parameter to make it natural
    a = rt.return_xpsd_xyzcool_yz_cool(folder_HP, xyz, yz, NFFT)
    xDg = a[0][2]
    freq = a[0][0]
    xpsd2_NOxFB = a[1][1]
    xpsd2_xFB = a[0][1]

    index0 = np.where( freq >= freq_fit_min )[0][0]
    index1 = np.where( freq >= freq_fit_max )[0][0]
    fit_points1 = np.logical_and(freq > freq_fit_min, freq < 59.0)
    fit_points2 = np.logical_and(freq > 61.0, freq < freq_fit_max)
    fit_points = fit_points1 + fit_points2

    poptNO, pcovNO = opt.curve_fit(psd2, freq[fit_points], xpsd2_NOxFB[fit_points], p0 = [106.8,  72.46,  41.97])

    p = list(poptNO) + [730., 1800., 1]

    bounds=([poptNO[0]-100, poptNO[1]-10, poptNO[2]-1, cut_low-0.1, cut_high -0.1, 0], [poptNO[0]+100, poptNO[1]+10, poptNO[2]+1, cut_low+0.1, cut_high +0.1, 10000])
    
    poptFB, pcovFB = opt.curve_fit(psd2_HP_feedback, freq[fit_points], xpsd2_xFB[fit_points], p0 = p, bounds = bounds)

    plt.figure()
    plt.loglog(freq, xpsd2_xFB)
    plt.loglog(freq, xpsd2_NOxFB)
    plt.loglog(freq, psd2(freq, *poptNO))
    plt.loglog(freq, psd2_HP_feedback(freq, *poptFB))

    # print "natural xDg = ", poptFB[5]
    # print "FPGA xDg = ", xDg
    # print "Gamma/2pi = ", poptNO[2]

    plt.xlim(20,200)
    plt.ylim(2e-5,1e-1)

    scale = poptFB[5]/xDg
    
    return scale

# extract_feedback_scale(folder_HP, xyz, yz, NFFT)

def Lambda(mass, gas_temp, gamma):
    return np.sqrt( 2.0*kb*gas_temp*gamma/mass)


def temperatude_folder_HP(folder_temp):
    name_load = str(folder_temp) + "\info.npy"
    data = np.load(name_load)
    
    freqHP = data[0]
    xpsd2_m_HP = data[2]
    
    index0 = np.where( freqHP >= freq_fit_min )[0][0]
    index1 = np.where( freqHP >= freq_fit_max )[0][0]
    
    fit_points1 = np.logical_and(freqHP > freq_fit_min, freqHP < 59.0)
    fit_points2 = np.logical_and(freqHP > 61.0, freqHP < freq_fit_max)
    fit_points = fit_points1 + fit_points2
    
    popt, pcov = opt.curve_fit(psd2, freqHP[fit_points], xpsd2_m_HP[fit_points], p0 = [0.01, 85, 10])
    
    if plotHP:
        plt.figure()
        plt.loglog(freqHP[index0:index1], xpsd2_m_HP[index0:index1])
        plt.loglog(freqHP, psd2(freqHP, *popt))
    
    df = freqHP[1] - freqHP[0]
    total_displacement2 = np.sum(psd2(freqHP, *popt)[index0:index1])*df
    
    spring_constant = mass*((2.*np.pi*popt[1])**2)
    
    mean_energy_spring = 0.5*spring_constant*total_displacement2
    
    temp = 2.0*mean_energy_spring/kb
    
    return [temp, popt[1]]




def temperatude_folder_nosphere(folder, freq, Diameter): # for no sphere, the temp comes from the area of the data, not the fit
    name_load = str(folder) + "\info.npy"
    data = np.load(name_load)
    
    freqLP = data[1]
    xpsd2_m_LP = data[3]
    
    xDg = data[6]
    
    index0 = np.where( freqLP >= freq_fit_min )[0][0]
    index1 = np.where( freqLP >= freq_fit_max )[0][0]

    if Diameter == 22.8e-6:
        fit_points1 = np.logical_and(freqLP > freq_fit_min, freqLP < 54.1)
        fit_points2 = np.logical_and(freqLP > 62.3, freqLP < 68.4)
        fit_points3 = np.logical_and(freqLP > 70.6, freqLP < 71.9)
        fit_points4 = np.logical_and(freqLP > 73.6, freqLP < 78.7)
        fit_points5 = np.logical_and(freqLP > 81.9, freqLP < 82.7)
        fit_points6 = np.logical_and(freqLP > 84.6, freqLP < 88.5)
        fit_points7 = np.logical_and(freqLP > 90.5, freqLP < 117.3)
        fit_points8 = np.logical_and(freqLP > 122.5, freqLP < 124.2)
        fit_points9 = np.logical_and(freqLP > 124.1, freqLP < freq_fit_max)
    
        fit_points = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8 + fit_points9
    
    if Diameter == 10.0e-6:
        fit_points1 = np.logical_and(freqLP > freq_fit_min, freqLP < 39.9)
        fit_points2 = np.logical_and(freqLP > 41.2, freqLP < 42.6)
        fit_points3 = np.logical_and(freqLP > 43.9, freqLP < 56.5)
        fit_points4 = np.logical_and(freqLP > 57.6, freqLP < 58.2)
        fit_points5 = np.logical_and(freqLP > 60.6, freqLP < 69.4)
        fit_points6 = np.logical_and(freqLP > 70.3, freqLP < 71.9)
        fit_points7 = np.logical_and(freqLP > 73.1, freqLP < 78.9)
        fit_points8 = np.logical_and(freqLP > 81.6, freqLP < 83.5)
        fit_points9 = np.logical_and(freqLP > 88.8, freqLP < 89.7)
        fit_points10 = np.logical_and(freqLP > 90.4, freqLP < 92.9)
        fit_points11 = np.logical_and(freqLP > 93.8, freqLP < freq_fit_max)
    
        fit_points = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8 + fit_points9 + fit_points10 + fit_points11
        
    if plotnosphere:
        plt.figure()
        plt.loglog(freqLP[index0:index1], xpsd2_m_LP[index0:index1])
    
    df = freqLP[1] - freqLP[0]
    total_displacement2 = np.sum(xpsd2_m_LP[fit_points])*df
    
    spring_constant = mass*((2.*np.pi*freq)**2)
    
    mean_energy_spring = 0.5*spring_constant*total_displacement2
    
    temp = 2.0*mean_energy_spring/kb
    
    return [temp, xDg, freqLP, xpsd2_m_LP]



def X_noise(freq, noise_psd2):
    Xnoise2 = noise_psd2*(freq[1]-freq[0])
    Xnoise = np.sqrt(Xnoise2)
    return Xnoise


def max_for_temp(freq, f_res ,natural_xDg, gamma, cut_low, cut_high, Xnoise, mass, gas_temp): # gets gamma/2pi evaluates the temperature considering possible squashing

    gamma = gamma
    f_res = f_res
    freq = freq
    
    a = 1.0*(f_res**4)*(natural_xDg**2)*feedback_filter_squared(freq, cut_low, cut_high)*Xnoise**2
    b = 1.0*((f_res**2 - freq**2)**2 + (gamma*freq)**2)*Xnoise**2

    L = Lambda(mass, gas_temp, gamma)

    c = 2.*L*(f_res**2)*(natural_xDg)*(feedback_filter_real(freq, cut_low, cut_high))*Xnoise
    d = 2.*L*(f_res**2 - freq**2)*Xnoise

    e = L**2 + c + a
    f = L**2 + d + b

    g = e/f
    g = max_between_array_ones(g)

    return g


# print max_for_temp(freq, 70., 50000.*2., 0.09, 730., 1800., Xnoise, mass, 50000.0)


def temperatude_folder(folder, freq, fit_vs_psd, Diameter, scale, gamma, cut_low, cut_high, Xnoise, mass, gas_temp):
    name_load = str(folder) + "\info.npy"
    data = np.load(name_load)
    
    freqLP = data[1]
    xpsd2_m_LP = data[3]
    
    xDg = data[6]
    
    index0 = np.where( freqLP >= freq_fit_min )[0][0]
    index1 = np.where( freqLP >= freq_fit_max )[0][0]

    if Diameter == 22.8e-6:
        fit_points1 = np.logical_and(freqLP > freq_fit_min, freqLP < 54.1)
        fit_points2 = np.logical_and(freqLP > 62.3, freqLP < 68.4)
        fit_points3 = np.logical_and(freqLP > 70.6, freqLP < 71.9)
        fit_points4 = np.logical_and(freqLP > 73.6, freqLP < 78.7)
        fit_points5 = np.logical_and(freqLP > 81.9, freqLP < 82.7)
        fit_points6 = np.logical_and(freqLP > 84.6, freqLP < 88.5)
        fit_points7 = np.logical_and(freqLP > 90.5, freqLP < 117.3)
        fit_points8 = np.logical_and(freqLP > 122.5, freqLP < 124.2)
        fit_points9 = np.logical_and(freqLP > 124.1, freqLP < freq_fit_max)
        
        fit_points = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8 + fit_points9
    
    if Diameter == 10.0e-6:
        fit_points1 = np.logical_and(freqLP > freq_fit_min, freqLP < 39.9)
        fit_points2 = np.logical_and(freqLP > 41.2, freqLP < 42.6)
        fit_points3 = np.logical_and(freqLP > 43.9, freqLP < 56.5)
        fit_points4 = np.logical_and(freqLP > 57.6, freqLP < 58.2)
        fit_points5 = np.logical_and(freqLP > 60.6, freqLP < 69.4)
        fit_points6 = np.logical_and(freqLP > 70.3, freqLP < 71.9)
        fit_points7 = np.logical_and(freqLP > 73.1, freqLP < 78.9)
        fit_points8 = np.logical_and(freqLP > 81.6, freqLP < 83.5)
        fit_points9 = np.logical_and(freqLP > 88.8, freqLP < 89.7)
        fit_points10 = np.logical_and(freqLP > 90.4, freqLP < 92.9)
        fit_points11 = np.logical_and(freqLP > 93.8, freqLP < freq_fit_max)
        
        fit_points = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8 + fit_points9 + fit_points10 + fit_points11
    
    try:
        popt, pcov = opt.curve_fit(psd2, freqLP[fit_points], xpsd2_m_LP[fit_points], p0 = [0.01, freq, 10])
        ff = True
    except:
        popt = [0.0003, 84.8, 0.08]
        print "plot failed" + str(folder)
        ff = False
        
    if plotLP:
        plt.figure()
        plt.loglog(freqLP[index0:index1], xpsd2_m_LP[index0:index1])
        plt.loglog(freqLP, psd2(freqLP, *popt))

    if not fit_vs_psd:
        ff = False
    df = freqLP[1] - freqLP[0]

    natural_xDg = scale*np.abs(xDg)

    
    maxi = max_for_temp(freqLP, freq ,natural_xDg, gamma, cut_low, cut_high, Xnoise, mass, gas_temp)

    if ff:
        total_displacement2_real = np.sum(maxi[fit_points]*psd2(freqLP, *popt)[fit_points])*df
        total_displacement2 = np.sum(psd2(freqLP, *popt)[fit_points])*df
        fres = popt[1]
    else:
        total_displacement2_real = np.sum(maxi[fit_points]*xpsd2_m_LP[fit_points])*df
        total_displacement2 = np.sum(xpsd2_m_LP[fit_points])*df
        fres = freq
    
    spring_constant = mass*((2.*np.pi*fres)**2)
    
    mean_energy_spring_real = 0.5*spring_constant*total_displacement2_real
    mean_energy_spring = 0.5*spring_constant*total_displacement2
    
    temp_real = 2.0*mean_energy_spring_real/kb
    temp = 2.0*mean_energy_spring/kb
    
    return [temp_real, temp, xDg]







[HP, fres] = temperatude_folder_HP(folder_list[0])

scale = extract_feedback_scale(folder_HP, xyz, yz, NFFT)

nosphere = temperatude_folder_nosphere(folder_nosphere, fres, Diameter)

Xnoise = X_noise(nosphere[2], nosphere[3])

Treal = []
T = []
D = []

for i in folder_list:
    a = temperatude_folder(i, fres, fit_true_psd_false, Diameter, scale, gammaover2pi, 730., 1800., Xnoise, mass, T0)
    Treal.append(a[0])
    T.append(a[1])
    D.append(np.abs(a[2])+1e-6)
plt.figure()
plt.loglog(D, 1.0e6*np.array(Treal), "ro")
plt.loglog(D, 1.0e6*np.array(T), "bo")
plt.hlines(1e6*nosphere[0], np.min(D)-2e-3, np.max(D)+2e-3)
plt.hlines(1e6*HP, np.min(D)-2e-3, np.max(D)+2e-3)
plt.grid()

plt.show()
