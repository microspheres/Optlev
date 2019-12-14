import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import glob
import scipy.optimize as opt


# freqhp, freq_meas, xpsd2_m_hp, xpsd2_m_meas, xpsd2_v_hp, xpsd2_meas_v, xDg


folder_temp = r"C:\data\20191107\22um\5\temp_x\14"
folder_nosphere = r"C:\data\20191107\22um\5\nosphere"

trans_out = (42./52.)**2
trans_in = (41./70.)**2 # this is the square of the power at the photodiode with sphere in and out.

# freq_plot_min = 60.
# freq_plot_max = 90.

freq_fit_min = 60.
freq_fit_max = 110.

# freq_fit_min =61.
# freq_fit_max = 90.



def psd(f, A, f0, gamma):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    gamma1 = 2.0*np.pi*gamma
    s1 = 2.*A*(gamma1*(w0**2))
    s2 = 1.*(w0**2)*((w0**2 - w**2)**2 + (gamma1*w)**2)
    s = np.sqrt(s1/s2)
    return s




def psd2(f, A, f0, gamma):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    gamma1 = 2.0*np.pi*gamma
    s1 = 2.*A*(gamma1*(w0**2))
    s2 = 1.*(w0**2)*((w0**2 - w**2)**2 + (gamma1*w)**2)
    s = np.sqrt(s1/s2)
    return (s)**2



def noise_psd(folder_nosphere, loop):
    if loop == "out":
        name_load = str(folder_nosphere) + "\info_outloop.npy"
    else:
        name_load = str(folder_nosphere) + "\info_inloop.npy"
    data = np.load(name_load)

    freqLP = data[1]
    xpsd2_m_LP = data[3]

    return [freqLP, xpsd2_m_LP]

    

def fit(folder, loop, noise_psd2):
    if loop == "out":
        noise_psd2 = trans_out*noise_psd2
        name_load = str(folder) + "\info_outloop.npy"
    else:
        noise_psd2 = trans_in*noise_psd2
        name_load = str(folder) + "\info_inloop.npy"
        
    data = np.load(name_load)

    
    freqLP = data[1]
    xpsd2_m_LP = data[3]
    xpsd2_m_LP = xpsd2_m_LP - noise_psd2
    
    xDg = data[6]
    
    index0 = np.where( freqLP >= freq_fit_min )[0][0]
    index1 = np.where( freqLP >= freq_fit_max )[0][0]

    fit_points1 = np.logical_and(freqLP > freq_fit_min, freqLP < 56.5)
    fit_points2 = np.logical_and(freqLP > 60.6, freqLP < 64.8)
    fit_points3 = np.logical_and(freqLP > 65.5, freqLP < 72.3)
    fit_points4 = np.logical_and(freqLP > 73.0, freqLP < 80.0)
    fit_points5 = np.logical_and(freqLP > 81.5, freqLP < 83.3)
    fit_points6 = np.logical_and(freqLP > 84.2, freqLP < 99.)
    fit_points7 = np.logical_and(freqLP > 102, freqLP < 108.8)
    fit_points8 = np.logical_and(freqLP > 109.5, freqLP < freq_fit_max)
    
    fit_points = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8
    
    popt, pcov = opt.curve_fit(psd2, freqLP[fit_points], xpsd2_m_LP[fit_points], p0 = [6e-7, 89.9, 0.000002])

    return [popt, pcov, freqLP, xpsd2_m_LP, noise_psd2]

b = noise_psd(folder_nosphere, "out")
a = fit(folder_temp, "out", b[1])


plt.figure()
plt.semilogy(a[2], a[3])
plt.semilogy(a[2], a[4])
plt.semilogy(a[2], psd2(a[2], *a[0]))
plt.semilogy(a[2], psd2(a[2], *[6e-7, 89.9, 0.000002]))
plt.xlim(10, 120)
plt.show()
