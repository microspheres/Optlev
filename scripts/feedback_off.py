import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt
import multiprocessing as mp
from scipy.signal import butter, lfilter, filtfilt

################# THIS CODE REQUIRES THE TRIGGER AND FEEDBACK ON AND OFF

############# THIS CODE REQUIRES THAT THE FIRST OFF EVENT DIDNT RAIL (THE FPGA DONT ACTIVATE THE FEEDBACK)

path_list = [r"C:\data\20190912\prechamber\9\onoff", ]


path_list = [r"C:\data\20190923\prechamber\3\Lowcharge\ONOFF2", ]

path_list = [r"C:\data\20190923\prechamber\3\onoff3"]

path_list = [r"C:\data\20191004\22um\prechamber_ATM\7\ONOFF4"]
# path_list = [r"C:\data\20191004\22um\prechamber\1\ONOFF4"]
path_list = [r"C:\data\20191014\22um\prechamber_ATM\3\ONOFF3"]
path_list = [r"C:\data\20191014\22um\prechamber_ATM\3\ONOFF5"]
path_list = [r"C:\data\20191014\22um\prechamber_LP\1\ONOFF"]
path_list = [r"C:\data\20191014\22um\prechamber_LP\5\ONOFF"]
path_list = [r"C:\data\20191014\22um\prechamber_LP\5\ONOFF2"]
path_list = [r"C:\data\20191014\22um\prechamber_LP\5\ONOFF3"]
path_list = [r"C:\data\20191014\22um\prechamber_LP\5\ONOFF4"]
path_list = [r"C:\data\20191014\22um\prechamber_LP\5\ONOFF5"]
path_list = [r"C:\data\paper\10um\PreChamber_LP\2\ONOFF\2"]
path_list = [r"C:\data\paper2\22um\PreChamber_LP\2\ONOFF\7"]
path_list = [r"C:\data\paper2\22um\PreChamber_LP\3\ONOFF13"]
path_list = [r"C:\data\paper3\22um\PreChamber_ATM\2\ONOFF_for_calibration"]
path_list = [r"C:\data\paper3\22um\PreChamber_ATM\2\ONOFF\5"]

#path_list = [r"C:\data\201908020\22um_SiO2_pinhole\5\ONOFF\1", ]

N = 15 # number of div

channel = bu.xi

plot_temp = True
v2_to_m2 = 1.122e-14
Diameter = 22.6e-6
rho = 1800.0

bins = 8
BPfilter = True
f1 = 10.
f2 = 300.
order = 4
gaussplot = True

kb = 1.380e-23

def mass(Diameter, rho):
    m = (4/3.)*(np.pi)*((Diameter/2)**3)*rho
    return m

mass = mass(Diameter, rho)

def temp(v2, mass, fres, v2_to_m2):
    t = v2*v2_to_m2*mass*(2.*np.pi*fres)**2
    return t/kb
    


def butter_lowpass(lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a


def butter_lowpass_filter(data, lowcut, fs, order):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def getdata(fname, axe):
	print "Opening file: ", fname
	## guess at file type from extension
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		#max_volt = dset.attrs['max_volt']
		#nbit = dset.attrs['nbit']
		Fs = dset.attrs['Fsamp']
		# Press = dset.attrs['pressures'][0]
                # print Press
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                PID = dset.attrs['PID']
                # print PID
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	x = dat[:, axe]-numpy.mean(dat[:, axe])
        if BPfilter:
                x = butter_bandpass_filter(x, f1, f2, Fs, order)
                # plt.figure()
                # plt.plot(x)
                # plt.show()
        trigger = dat[:, 4]

	return [x, trigger, PID, Fs]
    

Q = getdata(glob.glob((path_list[0]+"\*.h5"))[0], bu.xi)
fs = Q[3]
# plt.figure()
# t = np.array(range(len(Q[0])))/fs
# plt.plot(t,0.1*Q[1], label = "Trigger")
# plt.plot(t,Q[0], label = "Signal X direction")
# plt.xlabel("Time [s]")
# plt.ylabel("Signal [V]")
# plt.legend(loc=3)
# plt.tight_layout(pad = 0)
# plt.show()

def get_files_path(path):
    file_list = glob.glob(path+"\*.h5")
    return file_list

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

def get_data_path(path, axe):
    A = []
    for i in list_file_time_order(get_files_path(path)):
        a = getdata(i, axe)
        A.append(a)
    return A

def trigger_on2(path, a): # return index of PID ON
    F = get_data_path(path, a)
    indexf = []
    for i in range(len(F)):
        index = []
        indx = np.where((F[i][1]) > 3.)
        for j in range(len(indx[0])-1):
            b = (indx[0][j+1] - indx[0][j])
            if b != 1:
                indx2 = float(indx[0][j+1])
                index.append(indx2)
        indexf.append(np.array(index))
    return indexf

def trigger_off2(path, a): # return index of PID Off
    F = get_data_path(path, a)
    indexf = []
    for i in range(len(F)):
        index = []
        indx = np.where((F[i][1]) < 3.)
        index.append(indx[0][0])  # this is necessary due to the fact that the file is saved with the first trigger ON
        for j in range(len(indx[0])-1):
            b = (indx[0][j+1] - indx[0][j])
            if b != 1:
                indx2 = float(indx[0][j+1])
                index.append(indx2)
        indexf.append(np.array(index))
    return indexf

# print trigger_off2(path_list[0])

def line(x, a, b):
    return a*x + b

def gauss(x,a,b,c):
    g = c*np.exp(-0.5*((x-a)/b)**2)
    return g

def expo(x, A, B, C):
    return A*np.exp(B*x) + C

def expo2(x, A, B):
    return A*np.exp(B*x)


def chunk(timestream, number):
    l = len(timestream)
    n = int(l/number)
    a = 0
    New = []
    Time = []
    for i in range(number):
        t = np.mean(timestream[0+a:n+a-1])
        New.append(t)
        Time.append(a + int(n/2))
        a = a + n
    return [np.array(New), np.array(Time)]
        


def get_fb_off(path, axe, plot_temp, v2_to_m2, mass, freq_res):
    
    X = []
    timesoff = trigger_off2(path, axe)[0][:]
    timeson = trigger_on2(path, axe)[0][:]
    A = get_data_path(path, axe)
    
    for i in range(len(get_files_path(path))):
        x = A[i][0]
        X.append(x)

    Xall = [] # Xall is the data period of all times the feedback was OFF in the X sensor. This includes all files inside a folder.
    for i in X:
        for j in range(len(timesoff)):
            if j < len(timesoff) or j < len(timeson):
                xall = i[int(timesoff[j]):int(timeson[j])] - np.mean(i[int(timesoff[j]):int(timesoff[j])+1500])
                # plt.figure()
                # plt.plot(xall)
                # plt.show()
                Xall.append(xall)

    # this part now gets it squared

    Xallnew = []
    for i in Xall:
        xall = i[0:len(Xall[0])-100]
        if len(xall) == len(Xall[0])-100:
            # plt.figure()
            # plt.plot(i)
            # plt.show()
            Xallnew.append(xall)
    
    time = range(len(Xallnew[0]))/fs
    time2 = chunk(time, N)[0]

    heat = []
    outlier = []
    for i in Xallnew:
        a = i**2
        a = chunk(a, N)[0]
        p0 = np.array([0.004, 0.05])
        try:
            popt, pcov = opt.curve_fit(expo2, time2, a, p0 = p0)
            # print popt
            # plt.figure()
            # plt.plot(time2, a)
            # plt.plot(time2, expo2(time2, *popt))
            # plt.show()
            heat.append(popt[1]/(2.*np.pi))
            outlier.append(popt[1]/(2.*np.pi))
        except:
            print "individual fit failed"
            outlier.append(1e6)

    print "number of points = ", len(heat)
    print "mean heat = ", np.mean(heat)

    if gaussplot:
        h,b = np.histogram(heat, bins = bins)
        bc = np.diff(b)/2 + b[:-1]
        p0 = np.array([np.mean(heat), np.std(heat)/np.sqrt(len(heat)), 10])
        poptd, pcovd = opt.curve_fit(gauss, bc, h)

        space = np.arange(np.min(heat)-0.01, np.max(heat)+0.01, 0.0001)
    
        plt.figure()
        center = "%.2E"% poptd[0]
        error_ = "%.1E"% np.sqrt(pcovd[0][0])
        error = center + "$\pm$" + error_
        plt.plot(space, gauss(space, *poptd), label = error)
        plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = "ro")
        plt.ylabel("Counts")
        plt.xlabel("$\Gamma / 2\Pi$ [Hz]")
        plt.legend()
        plt.tight_layout(pad = 0)
        plt.grid()
        # plt.show()

    X2sum = np.zeros(len(Xallnew[0]))
    lenght_ = 0
    for i in range(len(Xallnew)):
        if True:
            x2 = Xallnew[i]**2
            X2sum = X2sum + x2
            lenght_ = lenght_ + 1
    X2sum = X2sum/lenght_

    f = chunk(X2sum, N)[0]

    p1 = np.array([0.004, 0.05])
    popt, pcov = opt.curve_fit(expo2, time2, f, p0 = p1)
    print "Gamma/2pi = ", popt[1]/(2.*np.pi)

    fig, ax = plt.subplots()
    ax.plot(time, X2sum)
    ax.plot(time2, f, "ro")
    ax.plot(time, expo2(time, *popt))
    if plot_temp:
        ax2 = ax.twinx()
        t1 = temp(X2sum[0], mass, freq_res, v2_to_m2)
        t2 = temp(X2sum[-1], mass, freq_res, v2_to_m2)
        ax2.set_ylim(t1, t2)
        ax.set_ylim(X2sum[0], X2sum[-1])
        ax2.set_ylabel("COM Temperature [K]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$<volt>^2$")
    plt.tight_layout(pad = 0)
    plt.grid()
    plt.show()

    # plt.figure()
    # plt.plot(time, X2sum)
    # plt.plot(time2, f, "ro")
    # plt.plot(time, expo2(time, *popt))
    # if plot_temp:
    ##      t = temp(f, mass, freq_res, v2_to_m2)
    # #     plt.axes.secondary_yaxis('right', functions=t)
    # plt.xlabel("Time [s]")
    # plt.ylabel("$<x>^2$")
    # plt.tight_layout(pad = 0)
    # plt.grid()
    # plt.show()

    
get_fb_off(path_list[0], channel, plot_temp, v2_to_m2, mass, 85.0)
