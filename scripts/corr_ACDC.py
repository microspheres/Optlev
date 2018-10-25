import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import bead_util as bu
import glob
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import cPickle as pickle

from correlation_each_freq_of_comb_main import *

steps = "False"
transfer = "False" # enable the transfer function, does not use correlation
calibration_mode = "False"
force_acc = "False" # to be used with a measurement with no ac field

mass = (2.58*10**-12) # in kg

Number_of_e = (7.76*10**14)

distance = 0.0021 # meters

V_calibration = 0.5 #as shown on the daq
V_meas_ac = 5.0 # as shown on the daq
V_max_ac = 20.0 # used on the force and acceleration... as shown on daq


freq_list = [48.]

path_charge = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\electrode_dist2_farther\calibration_1e"

path_signal = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\electrode_dist2_farther\meas2_ACDC"

path_noise = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\electrode_dist2_farther\meas2_ACDC"

path_save = path_signal

endfile = -1

startfile = 0

start_index = 0
end_index = -1

file_list_signal = glob.glob(path_signal+"\*.h5")
file_list_charge = glob.glob(path_charge+"\*.h5")
file_list_noise = glob.glob(path_noise+"\*.h5")

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

file_list_signal = list_file_time_order(file_list_signal)

file_list_signal = file_list_signal[startfile:endfile]

# DC_list = [-2000., -1429., -857., -286., 286., 857., 1429., 2000.]


# DC_list = [-5000., -3333., -1667., 0., 1667., 3333., 5000.]

#DC_list = [-7500., -5357., -3214., -1071., 1071., 3214., 5357., 7500.]

DC_list = [-7500., -5625., -3750., -1875., 0., 1875., 3750., 5625., 7500.]



def several_DC_list(file_list_signal, DC_list):
    L = [[] for i in range(len(DC_list))] # list of list

    for v in range(len(DC_list)):
        for i in range(len(file_list_signal)):
            a = float(re.findall("-?\d+mVdc",file_list_signal[i])[0][:-4])
            if DC_list[v] == a:
                b = file_list_signal[i]
                L[v].append(b)
    return L



def get_corr_DC_list(file_list_signal, file_list_charge, DC_list, fitgauss):

    d = drive(file_list_charge,1)
    arg = arg_each_freq(d, freq_list)
    xt = xtemplate_charge(file_list_charge,arg)
    jpsd_arg = jnoise(file_list_noise, arg)
    cali = auto_calibration(xt, jpsd_arg)

    L = several_DC_list(file_list_signal, DC_list)
    
    A = [[] for i in range(len(DC_list))]
    Aerror = [[] for i in range(len(DC_list))]
    
    for i in range(len(DC_list)):
        corr1 = corr(xt, L[i], jpsd_arg, arg)
        corrfreq = corr_allfreq(corr1)
        corrfreq = np.array(corrfreq)
        a = np.real(corrfreq*cali)
        if fitgauss:
            s = np.real(corrfreq*cali)

            def gauss(x,a,b,c):
                g = c*np.exp(-0.5*((x-a)/b)**2)
                return g

            h,b = np.histogram(s, bins = bins)

            bc = np.diff(b)/2 + b[:-1]

            p0 = [np.mean(s), np.std(s)/np.sqrt(len(L[i])), 2]

            try:
                popt, pcov = curve_fit(gauss, bc, h, p0)
            except:
                popt = p0
                pcov = np.zeros([len(p0),len(p0)])
            A[i].append(popt[0])
            Aerror[i].append(pcov[0][0])
            
            label_plot = str(popt[0]) + " $\pm$ " + str(np.sqrt(pcov[0,0]))
            space = np.linspace(bc[0],bc[-1], 1000)

            plt.figure()
            plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = 'ko', label = label_plot)
            plt.plot(space, gauss(space,*popt))
            plt.xlabel("Electron Number")
            plt.legend()
            plt.grid()
            
        else:
            a = np.mean(a)
            A[i].append(a)
            Aerror[i].append(0*a)

    return [A, Aerror]

def plot_corr_DC(file_list_signal, file_list_charge, DC_list, fitgauss, fitline):

    A, Aerror = get_corr_DC_list(file_list_signal, file_list_charge, DC_list, fitgauss)
    A = np.ndarray.flatten(np.array(A))
    Aerror = np.ndarray.flatten(np.array(Aerror))

    labelfit = ""

    if fitline:
        def line(x,a,b):
            return a*x + b
        popt, pcov = curve_fit(line, (200/1000.)*np.array(DC_list), np.array(A), sigma = np.array(Aerror))
        volt = np.linspace(min((200/1000.)*np.array(DC_list)) - 100, max((200/1000.)*np.array(DC_list)) + 100, 10)
        labelfit = str(popt)

    plt.figure()
    if fitgauss:
        plt.errorbar((200/1000.)*np.array(DC_list), np.array(A), yerr = np.sqrt(np.array(Aerror)), fmt = "ro")
    else:
        plt.plot((200/1000.)*np.array(DC_list), np.array(A), "ro")
    if fitline:
        plt.plot(volt, line(volt, *popt), "r--", label = labelfit)
    plt.xlabel("DC voltage [V]")
    plt.ylabel("Correlation")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path_save,'corr_DC.pdf'))
    return 

fitgauss = True
fitline = True

plot_corr_DC(file_list_signal, file_list_charge, DC_list, fitgauss, fitline)
plt.show()

