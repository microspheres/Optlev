import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
from scipy.special import wofz
import numpy as np
import glob
import scipy.optimize as opt
from scipy.stats import levy_stable


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


folder_save = r"C:\data\20191122\10um\2\temp_x4"
folder_temp = r"C:\data\20191122\10um\2\temp_x4"
folder_nosphere = r"C:\data\20191122\10um\2\nosphere"

trans_out = (26./30.)**2
trans_in = (41./70.)**2 # this is the square of the power at the photodiode with sphere in and out.

plotHP = False
plotLP = True
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

freq_fit_min =40.
freq_fit_max = 100.

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


def psd3(f, A, f0, gamma, alpha):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    gamma = 2.0*np.pi*gamma

    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.abs(1.*A*np.real(wofz((( w - w0 ) + 1j*gamma)/sigma/np.sqrt(2))) / sigma/np.sqrt(2*np.pi))

def find_FWHM(freq, data):
    Hmax = np.amax(data)/2.
    argmax = np.argmax(data)

    f_max = freq[argmax]

    aux = (data - Hmax)**2

    argmin = np.argmin(aux)

    freq_hmax = freq[argmin]

    return np.abs(f_max - freq_hmax)



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
        fit_points2 = np.logical_and(freqLP > 60.6, freqLP < 64.8)
        fit_points3 = np.logical_and(freqLP > 65.5, freqLP < 72.3)
        fit_points4 = np.logical_and(freqLP > 73.0, freqLP < 80.0)
        fit_points5 = np.logical_and(freqLP > 81.5, freqLP < 83.3)
        fit_points6 = np.logical_and(freqLP > 84.2, freqLP < 99.)
        fit_points7 = np.logical_and(freqLP > 102, freqLP < 108.8)
        fit_points8 = np.logical_and(freqLP > 109.5, freqLP < freq_fit_max)
    
        fit_points = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8
    
    if Diameter == 10.0e-6:
        fit_points1 = np.logical_and(freqLP > freq_fit_min, freqLP < 40.4)
        fit_points2 = np.logical_and(freqLP > 40.8, freqLP < 48.3)
        fit_points3 = np.logical_and(freqLP > 49.1, freqLP < 50.2)
        fit_points4 = np.logical_and(freqLP > 50.6, freqLP < 54.6)
        fit_points5 = np.logical_and(freqLP > 54.9, freqLP < 57.1)
        fit_points6 = np.logical_and(freqLP > 57.4, freqLP < 57.7)
        fit_points7 = np.logical_and(freqLP > 58.2, freqLP < 59.2)
        fit_points8 = np.logical_and(freqLP > 59.6, freqLP < 59.8)
        fit_points9 = np.logical_and(freqLP > 60.1, freqLP < 61.15)
        fit_points10 = np.logical_and(freqLP > 61.36, freqLP < 65.5)
        fit_points11 = np.logical_and(freqLP > 65.7, freqLP < 66.7)
        fit_points12 = np.logical_and(freqLP > 67.0, freqLP < 72.0)
        fit_points13 = np.logical_and(freqLP > 73.0, freqLP < 79.3)
        fit_points14 = np.logical_and(freqLP > 79.5, freqLP < freq_fit_max)
        
    
        fit_points = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8 + fit_points9 + fit_points10 + fit_points11 + fit_points12 + fit_points13 + fit_points14
        
    if plotnosphere:
        plt.figure()
        plt.loglog(freqLP[index0:index1], xpsd2_m_LP[index0:index1])
    
    df = freqLP[1] - freqLP[0]
    total_displacement2 = np.sum(xpsd2_m_LP[fit_points])*df
    
    spring_constant = mass*((2.*np.pi*freq)**2)
    
    mean_energy_spring = 0.5*spring_constant*total_displacement2
    
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
        fit_points1 = np.logical_and(freqLP > freq_fit_min, freqLP < 40.4)
        fit_points2 = np.logical_and(freqLP > 40.8, freqLP < 48.3)
        fit_points3 = np.logical_and(freqLP > 49.1, freqLP < 50.2)
        fit_points4 = np.logical_and(freqLP > 50.6, freqLP < 54.6)
        fit_points5 = np.logical_and(freqLP > 54.9, freqLP < 57.1)
        fit_points6 = np.logical_and(freqLP > 57.4, freqLP < 57.7)
        fit_points7 = np.logical_and(freqLP > 58.2, freqLP < 59.2)
        fit_points8 = np.logical_and(freqLP > 59.6, freqLP < 59.8)
        fit_points9 = np.logical_and(freqLP > 60.1, freqLP < 61.15)
        fit_points10 = np.logical_and(freqLP > 61.36, freqLP < 65.5)
        fit_points11 = np.logical_and(freqLP > 65.7, freqLP < 66.7)
        fit_points12 = np.logical_and(freqLP > 67.0, freqLP < 72.0)
        fit_points13 = np.logical_and(freqLP > 73.0, freqLP < 79.3)
        fit_points14 = np.logical_and(freqLP > 79.5, freqLP < freq_fit_max)
        
    
        fit_points = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8 + fit_points9 + fit_points10 + fit_points11 + fit_points12 + fit_points13 + fit_points14

    try:
        popt, pcov = opt.curve_fit(psd2, freqLP[fit_points], xpsd2_m_LP[fit_points], p0 = [1e-5, freq, 10])
        print popt
    except:
        popt = [0.0003, 60.8, 0.08]
        print "plot failed" + str(folder)


    try:
        popt2, pcov2 = opt.curve_fit(psd3, freqLP[fit_points], xpsd2_m_LP[fit_points], p0 = [1e-15, 60.5, 0.05, 1e-1])
    except:
        popt2 = [1e-15, 60.5, 0.05, 1e-1]
        print "plot failed" + str(folder)


    r1 = np.sum((psd2(freqLP, *popt)[fit_points] - xpsd2_m_LP[fit_points])**2)
    r2 = np.sum((psd3(freqLP, *popt2)[fit_points] - xpsd2_m_LP[fit_points])**2)

    if r1 > r2:
        popt = popt2
        voigt = True
    else:
        popt = popt
        voigt = False
        
    if plotLP:
        plt.figure()
        label = str(xDg)+str(loop)+"_"+str(folder)
        plt.semilogy(freqLP[fit_points], xpsd2_m_LP[fit_points], label = label)
        if voigt:
            plt.semilogy(freqLP, psd3(freqLP, *popt))
        else:
            plt.semilogy(freqLP, psd2(freqLP, *popt))
        plt.xlim(freq_fit_min, freq_fit_max)
        plt.legend()

    df = freqLP[1] - freqLP[0]

    total_displacement2 = np.sum(xpsd2_m_LP[fit_points])*df
    fres = freq
    spring_constant = mass*((2.*np.pi*fres)**2)
    mean_energy_spring = 0.5*spring_constant*total_displacement2
    temp = 2.0*mean_energy_spring/kb
    
    if voigt:
        total_displacement2_fit = np.sum(psd3(freqLP[index0:index1], *popt))*df
        w = find_FWHM(freqLP[index0:index1], psd3(freqLP[index0:index1], *popt))

    else:
        total_displacement2_fit = np.sum(psd2(freqLP[index0:index1], *popt))*df
        w = find_FWHM(freqLP[index0:index1], psd2(freqLP[index0:index1], *popt))
 
    fres = freq
    spring_constant_fit = mass*((2.*np.pi*fres)**2)
    mean_energy_spring_fit = 0.5*spring_constant_fit*total_displacement2_fit
    temp_fit = 2.0*mean_energy_spring_fit/kb
    
    return [temp, xDg, temp_fit, w]




list_of_plots = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# freqhp, freq_meas, xpsd2_m_hp, xpsd2_m_meas, xpsd2_v_hp, xpsd2_meas_v, xDg
def selected_plot_outloop(folder_list, folder_nosphere, list_of_plots):
    
    name_load_nosphere = str(folder_nosphere) + "\info_outloop.npy"
    data_no_sphere = np.load(name_load_nosphere)

    f = data_no_sphere[1]
    xpsd_nosphere = np.sqrt(data_no_sphere[3]*trans_out)

    index0 = np.where( f >= freq_fit_min )[0][0]
    index1 = np.where( f >= freq_fit_max )[0][0]
    
    plt.figure()
    plt.semilogy(f[index0:index1], xpsd_nosphere[index0:index1], label = "no_sphere")
    plt.xlim(freq_fit_min, freq_fit_max)

    for i in range(len(folder_list)):
        if list_of_plots[i] == 1:
            name_load = str(folder_list[i]) + "\info_outloop.npy"
            data = np.load(name_load)
            f = data[1]
            xpsd = np.sqrt(data[3])
            plt.semilogy(f[index0:index1], xpsd[index0:index1])
        plt.legend()
        plt.ylabel("PSD X [m/$\sqrt{Hz}$]")
        plt.xlabel("Frequency [Hz]")
        plt.tight_layout(pad = 0)
        plt.title("outloop")
        plt.grid()
        name = "outloop_" + ".pdf"
        name = os.path.join(folder_save, name)
        # plt.savefig(name)
            
    return []

def selected_plot_inloop(folder_list, folder_nosphere, list_of_plots):
    
    name_load_nosphere = str(folder_nosphere) + "\info_inloop.npy"
    data_no_sphere = np.load(name_load_nosphere)

    f = data_no_sphere[1]
    xpsd_nosphere = np.sqrt(data_no_sphere[3]*trans_in)

    index0 = np.where( f >= freq_fit_min )[0][0]
    index1 = np.where( f >= freq_fit_max )[0][0]
    
    plt.figure()
    plt.semilogy(f[index0:index1], xpsd_nosphere[index0:index1], label = "no_sphere")
    plt.xlim(freq_fit_min, freq_fit_max)

    for i in range(len(folder_list)):
        if list_of_plots[i] == 1:
            name_load = str(folder_list[i]) + "\info_inloop.npy"
            data = np.load(name_load)
            f = data[1]
            xpsd = np.sqrt(data[3])
            plt.semilogy(f[index0:index1], xpsd[index0:index1])
        plt.legend()
        plt.ylabel("PSD X [m/$\sqrt{Hz}$]")
        plt.xlabel("Frequency [Hz]")
        plt.tight_layout(pad = 0)
        plt.title("inloop")
        plt.grid()
        name = "inloop_" + ".pdf"
        name = os.path.join(folder_save, name)
        # plt.savefig(name)
            
    return []

selected_plot_outloop(folder_list, folder_nosphere, list_of_plots)
selected_plot_inloop(folder_list, folder_nosphere, list_of_plots)

[HPout, fres] = temperatude_folder_HP(folder_list[0], "out")
[HPin, fres] = temperatude_folder_HP(folder_list[0], "in")
nosphere_out = temperatude_folder_nosphere(folder_nosphere, fres, Diameter, "out")
nosphere_in = temperatude_folder_nosphere(folder_nosphere, fres, Diameter, "in")

noise_in_psd2 = nosphere_in[2]
noise_out_psd2 = nosphere_out[2]


# nolaser_out = temperatude_folder_nosphere(folder_nolaser, fres, Diameter, "out")
# nolaser_in = temperatude_folder_nosphere(folder_nolaser, fres, Diameter, "in")

T = []
T_fit = []
D = []
W = []

T2 = []
T2_fit = []
D2 = []
for i in folder_list:
    a = temperatude_folder(i, fres, fit_true_psd_false, Diameter, "out", noise_out_psd2)
    T.append(a[0])
    T_fit.append(a[2])
    D.append(np.abs(a[1])+1e-6)
    W.append(a[3])

    a2 = temperatude_folder(i, fres, fit_true_psd_false, Diameter, "in", noise_in_psd2)
    T2.append(a2[0])
    T2_fit.append(a2[2])
    D2.append(np.abs(a2[1])+1e-6)

plt.figure()
plt.plot(D, W, "r.")
    
plt.figure()
plt.loglog(D, 1.0e6*np.array(T), "ro", label = "outloop")
plt.loglog(D, 1.0e6*np.array(T_fit), "rx", label = "outloop_fit")
plt.hlines(1e6*nosphere_out[0], np.min(D)-2e-3, np.max(D)+2e-3, color = "r", label = "no sphere outloop")
plt.hlines(1e6*HPout, np.min(D)-2e-3, np.max(D)+2e-3, label = "1mbar outloop")
plt.loglog(D, 1.0e6*np.array(T2), "bo", label = "inloop")
plt.loglog(D, 1.0e6*np.array(T2_fit), "bx", label = "inloop_fit")
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
