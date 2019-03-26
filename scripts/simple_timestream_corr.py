import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import os, re, time, glob
from scipy.signal import butter, lfilter, filtfilt
from scipy import signal
import bead_util as bu
from scipy.optimize import curve_fit
import glob


bins = 15 #hist bins!

order = 2

p = 3
p1 = bu.xi

v_cali = 0.6
v_meas = 2.6
Number_of_e = (8.4*10**14)

ind = 0
end = -1

path_charge = r"C:\data\20190322\15um_low532\1\1p_calibration"

path_signal = r"C:\data\20190322\15um_low532\1\charge6"

file_list_signal = glob.glob(path_signal+"\*.h5")
file_list_charge = glob.glob(path_charge+"\*.h5")

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

file_list_signal = list_file_time_order(file_list_signal)

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



def getdata_x_d(fname, ind, end):
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		Fs = dset.attrs['Fsamp']
		dat = dat * 10./(2**15 - 1)
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )
        
	x = dat[ind:end, p1]-numpy.mean(dat[ind:end, p1])
	drive = dat[ind:end, p] - numpy.mean(dat[ind:end, p])
        
        x = butter_bandpass_filter(x, 45, 51, Fs, order)
        drive = butter_bandpass_filter(drive, 45, 51, Fs, order)
	
	return [x, drive]

def corr(x, drive):
    x = x - np.mean(x)
    drive = drive - np.mean(drive)
    c = np.sum(x*drive)
    return c

def func(x,A,w,ph):
    return 1.0*A*np.sin(x*w + ph)

def drive(file_list_charge, ind, end):
    D = []
    for i in file_list_charge:
        x,d = getdata_x_d(i, ind, end)
        d = np.array(x) # use x as a a drive
        D.append(d)
    d = 0
    for i in range(len(file_list_charge)):
        d = D[i] + d

    drive = 1.0*d/len(file_list_charge)
    # p0 = np.array([0.5,1/33.86,0.01])
    #t = np.array(range(len(drive)))
    # popt, pcov = curve_fit(func, t, drive, p0 = p0)
    # print len(drive)
    #plt.figure()
    #plt.plot(t, drive)
    # print popt
    # plt.plot(t,func(t,*popt))
    return drive
 
def calibration(file_list_charge, ind, end): # outputs the correlation for 1 charge
    drives = drive(file_list_charge, ind, end)
    C = []
    for i in file_list_charge:
        x,d = getdata_x_d(i, ind, end)
        c = corr(x, drives)
        C.append(c)
    C = np.mean(np.array(C))
    return C

def gauss(x,a,b,c):
    g = c*np.exp(-0.5*((x-a)/b)**2)
    return g

def corr_gauss(file_list_signal, file_list_charge, ind, end, v_meas):
    cali = 1./calibration(file_list_charge, ind, end)
    cali = (v_cali/v_meas)/(Number_of_e)*cali
    drives = drive(file_list_charge, ind, end)
    C = []
    for i in file_list_signal:
        print "doing", i
        x,d = getdata_x_d(i, ind, end)
        c = corr(x,drives)
        C.append(c)
    C = np.array(C)*cali
    print "mean corr", np.mean(C)
    
    s = C
    plt.figure()
    plt.plot(s, "ro")
    plt.grid()
    plt.ylabel("correlation")

    h,b = np.histogram(s, bins = bins)

    bc = np.diff(b)/2 + b[:-1]

    p0 = [np.mean(s), np.std(s), len(s)/30.]
    try:
        popt, pcov = curve_fit(gauss, bc, h, p0)
    except:
        print "gauss fit failed"
        popt = p0
        pcov = np.zeros([len(p0),len(p0)])


    space = np.linspace(bc[0],bc[-1], 1000)

    label_plot = str(popt[0]) + " $\pm$ " + str(np.sqrt(pcov[0,0]))

    print "result from charge fit in e#"
    print popt
    print np.sqrt(pcov[0,0])

    plt.figure()
    plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = 'ko', label = label_plot)
    plt.plot(space, gauss(space,*popt))
    plt.xlabel("correlation Number of e")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path_signal,'histogram_corr_timestream.pdf'))
    return [popt[0], np.sqrt(pcov[0,0])]


def several_index(file_list_signal, file_list_charge, ind, end, v_meas):
    M = []
    E = []
    In = []
    step = 1.0 * end
    while ind < 2**19:
        m, e = corr_gauss(file_list_signal, file_list_charge, ind, end, v_meas)
        M.append(m)
        E.append(e)
        In.append(ind)
        ind = ind + step
        end = end + step
    plt.figure()
    plt.errorbar(In ,M, yerr = E, fmt = "ro--")
    plt.grid()


def organize_DC_pos(list):
    file_list_new = []
    for i in range(len(list)):
        if float(re.findall("-?\d+mVdc",list[i])[0][:-4]) > 0:
            file_list_new.append(list[i])
    return file_list_new

def organize_DC_neg(list):
    file_list_new = []
    for i in range(len(list)):
        if float(re.findall("-?\d+mVdc",list[i])[0][:-4]) < 0:
            file_list_new.append(list[i])
    return file_list_new


def corr_gauss_ACDC(file_list_signal, file_list_charge, ind, end):
    pos = organize_DC_pos(file_list_signal)
    neg = organize_DC_neg(file_list_signal)

    cali = 1./calibration(file_list_charge, ind, end)
    cali = (v_cali/v_meas)/(Number_of_e)*cali
    drives = drive(file_list_charge, ind, end)
    Cp = []
    Cn = []
    for i in pos:
        print "doing", i
        x,d = getdata_x_d(i, ind, end)
        c = corr(x,drives)
        Cp.append(c)
    Cp = np.array(Cp)*cali
    for i in neg:
        print "doing", i
        x,d = getdata_x_d(i, ind, end)
        c = corr(x,drives)
        Cn.append(c)
    Cn = np.array(Cn)*cali

    if len(Cp) > len(Cn):
        Cp = Cp[0:-1]

    if len(Cp) < len(Cn):
        Cn = Cn[0:-1]
    
    C = Cp + Cn

    plt.figure()
    plt.plot(C, "ro", label = "mean")
    plt.plot(Cp, "go", label = "pos")
    plt.plot(Cn, "bo", label = "neg")
    plt.legend()
    plt.grid()
    plt.ylabel("correlation")
    plt.savefig(os.path.join(path_signal,'corr_in_time.pdf'))
    plt.figure()
    plt.plot(C, "ro", label = "mean")
    plt.legend()
    plt.grid()
    plt.ylabel("correlation")

    print "mean corr", np.mean(C)
    
    s = C
    
    h,b = np.histogram(s, bins = bins)

    bc = np.diff(b)/2 + b[:-1]

    p0 = [np.mean(s), np.std(s)/np.sqrt(len(s)), len(s)/30.]
    try:
        popt, pcov = curve_fit(gauss, bc, h, p0)
    except:
        popt = p0
        pcov = np.zeros([len(p0),len(p0)])


    space = np.linspace(bc[0],bc[-1], 1000)

    label_plot = str(popt[0]) + " $\pm$ " + str(np.sqrt(pcov[0,0]))

    print "result from charge fit in e#"
    print popt
    print np.sqrt(pcov[0,0])

    plt.figure()
    plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = 'ko', label = label_plot)
    plt.plot(space, gauss(space,*popt))
    plt.xlabel("correlation Number of e")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path_signal,'histogram_corr_timestream.pdf'))
    return [popt[0], np.sqrt(pcov[0,0])]



#corr_gauss_ACDC(file_list_signal, file_list_charge, ind, end)
corr_gauss(file_list_signal, file_list_charge, ind, end, v_meas)

#several_index(file_list_signal, file_list_charge, 0, 12500, v_meas)


plt.show()
