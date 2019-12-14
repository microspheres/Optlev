import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import glob
import scipy.optimize as opt

# freqhp, freq_meas, xpsd2_m_hp, xpsd2_m_meas, xpsd2_v_hp, xpsd2_meas_v, xDg




folder_list = [r"C:\data\20191104\QFP\22um\2\temp_x_2\1", r"C:\data\20191104\QFP\22um\2\temp_x_2\2", r"C:\data\20191104\QFP\22um\2\temp_x_2\3",r"C:\data\20191104\QFP\22um\2\temp_x_2\4", r"C:\data\20191104\QFP\22um\2\temp_x_2\5", r"C:\data\20191104\QFP\22um\2\temp_x_2\6", r"C:\data\20191104\QFP\22um\2\temp_x_2\7",r"C:\data\20191104\QFP\22um\2\temp_x_2\8", r"C:\data\20191104\QFP\22um\2\temp_x_2\9",r"C:\data\20191104\QFP\22um\2\temp_x_2\10", r"C:\data\20191104\QFP\22um\2\temp_x_2\11",r"C:\data\20191104\QFP\22um\2\temp_x_2\12", r"C:\data\20191104\QFP\22um\2\temp_x_2\13", r"C:\data\20191104\QFP\22um\2\temp_x_2\14",r"C:\data\20191104\QFP\22um\2\temp_x_2\15",r"C:\data\20191104\QFP\22um\2\temp_x_2\16",]

folder_list = [r"C:\data\20191107\22um\2\temp_x\1", r"C:\data\20191107\22um\2\temp_x\2", r"C:\data\20191107\22um\2\temp_x\3", r"C:\data\20191107\22um\2\temp_x\4", r"C:\data\20191107\22um\2\temp_x\5", r"C:\data\20191107\22um\2\temp_x\6", r"C:\data\20191107\22um\2\temp_x\7", r"C:\data\20191107\22um\2\temp_x\8", r"C:\data\20191107\22um\2\temp_x\9", r"C:\data\20191107\22um\2\temp_x\10", r"C:\data\20191107\22um\2\temp_x\11", r"C:\data\20191107\22um\2\temp_x\12", r"C:\data\20191107\22um\2\temp_x\13", r"C:\data\20191107\22um\2\temp_x\14", r"C:\data\20191107\22um\2\temp_x\15", r"C:\data\20191107\22um\2\temp_x\16", r"C:\data\20191107\22um\2\temp_x\17", r"C:\data\20191107\22um\2\temp_x\18", r"C:\data\20191107\22um\2\temp_x\19", r"C:\data\20191107\22um\2\temp_x\20", r"C:\data\20191107\22um\2\temp_x\21", r"C:\data\20191107\22um\2\temp_x\22", r"C:\data\20191107\22um\2\temp_x\23", r"C:\data\20191107\22um\2\temp_x\24", r"C:\data\20191107\22um\2\temp_x\25", r"C:\data\20191107\22um\2\temp_x\26", r"C:\data\20191107\22um\2\temp_x\27", r"C:\data\20191107\22um\2\temp_x\28", r"C:\data\20191107\22um\2\temp_x\29", r"C:\data\20191107\22um\2\temp_x\30", r"C:\data\20191107\22um\2\temp_x\31", r"C:\data\20191107\22um\2\temp_x\32", r"C:\data\20191107\22um\2\temp_x\33"]

folder_list = [r"C:\data\20191107\22um\3\tempx_2\1", r"C:\data\20191107\22um\3\tempx_2\2", r"C:\data\20191107\22um\3\tempx_2\3",r"C:\data\20191107\22um\3\tempx_2\4",r"C:\data\20191107\22um\3\tempx_2\5", r"C:\data\20191107\22um\3\tempx_2\5_zstrong",]


folder_list = [r"C:\data\20191107\22um\5\temp_x\1", r"C:\data\20191107\22um\5\temp_x\2", r"C:\data\20191107\22um\5\temp_x\3", r"C:\data\20191107\22um\5\temp_x\4", r"C:\data\20191107\22um\5\temp_x\5", r"C:\data\20191107\22um\5\temp_x\6", r"C:\data\20191107\22um\5\temp_x\7", r"C:\data\20191107\22um\5\temp_x\8", r"C:\data\20191107\22um\5\temp_x\9", r"C:\data\20191107\22um\5\temp_x\10", r"C:\data\20191107\22um\5\temp_x\11", r"C:\data\20191107\22um\5\temp_x\12",r"C:\data\20191107\22um\5\temp_x\13",r"C:\data\20191107\22um\5\temp_x\14",r"C:\data\20191107\22um\5\temp_x\15",r"C:\data\20191107\22um\5\temp_x\16",r"C:\data\20191107\22um\5\temp_x\17", ]


folder_nosphere = r"C:\data\20191107\22um\5\nosphere"
# folder_nolaser = r"C:\data\20191104\QFP\22um\2\nolaser"

plotHP = True
plotLP = False
plotnosphere = False

fit_true_psd_false = False # how to calculate the energy

freq_plot_min = 30.
freq_plot_max = 130.

freq_fit_min = 60.
freq_fit_max = 110.

# freq_fit_min =45.
# freq_fit_max = 100.

Diameter = 22.8e-6 # meters
# Diameter = 10.0e-6
rho = 1800

kb = 1.380e-23

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
    s1 = 2.*A*(gamma1*(w0**2))
    s2 = 1.*(w0**2)*((w0**2 - w**2)**2 + (gamma1*w)**2)
    s = np.sqrt(s1/s2)
    return (A*s)**2



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
        fit_points2 = np.logical_and(freqLP > 61.1, freqLP < 64.8)
        fit_points3 = np.logical_and(freqLP > 66.2, freqLP < 71.2)
        fit_points4 = np.logical_and(freqLP > 72.6, freqLP < 79.0)
        fit_points5 = np.logical_and(freqLP > 79.8, freqLP < 80.7)
        fit_points6 = np.logical_and(freqLP > 81.6, freqLP < 83.0)
        fit_points7 = np.logical_and(freqLP > 84.4, freqLP < 88.4)
        fit_points8 = np.logical_and(freqLP > 89.3, freqLP < freq_fit_max)
    
        fit_points = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8
    
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
    
    return [temp, xDg]




def temperatude_folder(folder, freq, fit_vs_psd, Diameter, loop):
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
        fit_points2 = np.logical_and(freqLP > 61.1, freqLP < 64.8)
        fit_points3 = np.logical_and(freqLP > 66.2, freqLP < 71.2)
        fit_points4 = np.logical_and(freqLP > 72.6, freqLP < 79.0)
        fit_points5 = np.logical_and(freqLP > 79.8, freqLP < 80.7)
        fit_points6 = np.logical_and(freqLP > 81.6, freqLP < 83.0)
        fit_points7 = np.logical_and(freqLP > 84.4, freqLP < 88.4)
        fit_points8 = np.logical_and(freqLP > 89.3, freqLP < 117.7)
    
        fit_points = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8
    
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
    if ff:
        total_displacement2 = np.sum(psd2(freqLP, *popt)[index0:index1])*df
        fres = popt[1]
    else:
        total_displacement2 = np.sum(xpsd2_m_LP[fit_points])*df
        fres = freq
    
    spring_constant = mass*((2.*np.pi*fres)**2)
    
    mean_energy_spring = 0.5*spring_constant*total_displacement2
    
    temp = 2.0*mean_energy_spring/kb
    
    return [temp, xDg]



[HPout, fres] = temperatude_folder_HP(folder_list[0], "out")
[HPin, fres] = temperatude_folder_HP(folder_list[0], "in")
nosphere_out = temperatude_folder_nosphere(folder_nosphere, fres, Diameter, "out")
nosphere_in = temperatude_folder_nosphere(folder_nosphere, fres, Diameter, "in")


# nolaser_out = temperatude_folder_nosphere(folder_nolaser, fres, Diameter, "out")
# nolaser_in = temperatude_folder_nosphere(folder_nolaser, fres, Diameter, "in")

T = []
D = []

T2 = []
D2 = []
for i in folder_list:
    a = temperatude_folder(i, fres, fit_true_psd_false, Diameter, "out")
    T.append(a[0])
    D.append(np.abs(a[1])+1e-6)

    a2 = temperatude_folder(i, fres, fit_true_psd_false, Diameter, "in")
    T2.append(a2[0])
    D2.append(np.abs(a2[1])+1e-6)
    
plt.figure()
plt.loglog(D, 1.0e6*np.array(T), "ro", label = "outloop")
plt.hlines(1e6*nosphere_out[0], np.min(D)-2e-3, np.max(D)+2e-3, color = "r", label = "no sphere")
plt.hlines(1e6*HPout, np.min(D)-2e-3, np.max(D)+2e-3)
plt.loglog(D, 1.0e6*np.array(T2), "bo", label = "inloop")
plt.hlines(1e6*nosphere_in[0], np.min(D)-2e-3, np.max(D)+2e-3, color = "b", label = "no sphere")
plt.hlines(1e6*HPin, np.min(D2)-2e-3, np.max(D2)+2e-3)

# plt.hlines(1e6*nolaser_in[0], np.min(D)-2e-3, np.max(D)+2e-3, color = "b", label = "laser off", linestyle = "--")
# plt.hlines(1e6*nolaser_out[0], np.min(D)-2e-3, np.max(D)+2e-3, color = "r", label = "laser off", linestyle = "--")

plt.legend()
plt.grid()

plt.show()
