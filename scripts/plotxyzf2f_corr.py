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

bins = 15

Fs = 1e4

order = 2

voltage_scan = True # if true the label of the horizontal axis will become voltage (10kV trek only)

dont_calibrate = False # false for most use

fit_sub_plots = True  # when true it is nice to organize file list to be in order

use_list_charge = False

dist = False ## enables 1/cali plot on x axi

#motor 4 is horizontal; motor 2 is vertical

ind = 100000

v_cali = 0.15

v_meas = 12.0

Number_of_e = (7.76*10**14)

# j counts x,y,z,x2,y2,z2, i counts the signal index

list_of_charge_folders = [r"C:\data\20180702\bead2_SiO2_15um_POL_NS\distances\dist16\several_voltages\calibration1e"]

path_save = r"C:\data\20180702\bead2_SiO2_15um_POL_NS\distances\dist16\several_voltages\voltages3"

list_of_signal_folders = [r"C:\data\20180702\bead2_SiO2_15um_POL_NS\distances\dist16\several_voltages\voltages3\1",
                          r"C:\data\20180702\bead2_SiO2_15um_POL_NS\distances\dist16\several_voltages\voltages3\2",
                          r"C:\data\20180702\bead2_SiO2_15um_POL_NS\distances\dist16\several_voltages\voltages3\3",
                          r"C:\data\20180702\bead2_SiO2_15um_POL_NS\distances\dist16\several_voltages\voltages3\4",
                          r"C:\data\20180702\bead2_SiO2_15um_POL_NS\distances\dist16\several_voltages\voltages3\5",
                          r"C:\data\20180702\bead2_SiO2_15um_POL_NS\distances\dist16\several_voltages\voltages3\6",
                          r"C:\data\20180702\bead2_SiO2_15um_POL_NS\distances\dist16\several_voltages\voltages3\7",
                          r"C:\data\20180702\bead2_SiO2_15um_POL_NS\distances\dist16\several_voltages\voltages3\8",
                          r"C:\data\20180702\bead2_SiO2_15um_POL_NS\distances\dist16\several_voltages\voltages3\9",
                          r"C:\data\20180702\bead2_SiO2_15um_POL_NS\distances\dist16\several_voltages\voltages3\10",
                          r"C:\data\20180702\bead2_SiO2_15um_POL_NS\distances\dist16\several_voltages\voltages3\11",
                          r"C:\data\20180702\bead2_SiO2_15um_POL_NS\distances\dist16\several_voltages\voltages3\12",]


# in general you can write the following:
# tilt = np.array([0, 1])/6670.
# tilterr = np.array([0, 0])*1./(25.*25.4)

# tilt = np.array([1])
# tilterr = tilt*0

ll = len(list_of_signal_folders)
tilt = np.arange(1,ll+1) # arbitrary unit for distance for this plot
tilterr = np.ones(ll)*0

make_folder = np.ones(ll)


# make_folder = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


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


def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

def getdata(fname):
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
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

        x = dat[ind:, bu.xi]-numpy.mean(dat[ind:, bu.xi])
        y = dat[ind:, bu.yi]-numpy.mean(dat[ind:, bu.yi])
        z = dat[ind:, bu.zi]-numpy.mean(dat[ind:, bu.zi])
        d = dat[ind:, bu.drive]-numpy.mean(dat[ind:, bu.drive])

	return [x, y, z, d]

def calibration(file_list_charge): # outputs the correlation for 1 charge
    drives = Drive(file_list_charge)[0]
    C = []
    for i in file_list_charge:
        x = getdata(i)[0]
        x = butter_bandpass_filter(x, 45, 51, Fs, order)
        c = corr(x, drives)
        C.append(c)
    C = np.mean(np.array(C))
    return C


def Drive(file_list_charge):
    D = []
    for i in file_list_charge:
        x = getdata(i)[0]
        d = np.array(x) # use x as a a drive
        D.append(d)
    d = 0
    for i in range(len(file_list_charge)):
        d = D[i] + d

    drive = 1.0*d/len(file_list_charge)
    
    drive = butter_bandpass_filter(drive, 45, 51, Fs, order)

    drive2 = drive*drive

    return [drive, drive2]


def corr(x, drive):
    x = x - np.mean(x)
    drive = drive - np.mean(drive)
    c = np.sum(x*drive)
    return c


def gauss(x,a,b,c):
        g = c*np.exp(-0.5*((x-a)/b)**2)
        return g

def corr_in_folders(list_of_signal_folders, list_of_charge_folders):
    
    if len(list_of_signal_folders) != len(list_of_charge_folders) and use_list_charge:
        return "diff len of calibration and signal"
    
    L = [[] for i in range(len(list_of_signal_folders))] # list of list
    Cali = []

    for i in range(len(list_of_signal_folders)):

        if not use_list_charge:
            file_list_charge = glob.glob(list_of_charge_folders[0]+"\*.h5")
            drive1, drive2 = Drive(file_list_charge)
        else:
            file_list_charge = glob.glob(list_of_charge_folders[i]+"\*.h5")
            drive1, drive2 = Drive(file_list_charge)
            
        if not dont_calibrate:
            cali = 1./calibration(file_list_charge)
            Cali.append(1./cali)
            cali = (v_cali/v_meas)/(Number_of_e)*cali
        else:
            cali = 1.

        file_list_signal = glob.glob(list_of_signal_folders[i]+"\*.h5")
        file_list_signal = list_file_time_order(file_list_signal)
        if make_folder[i]:
            
            X = []
            Y = []
            Z = []
            X2 = []
            Y2 = []
            Z2 = []
            for j in file_list_signal:
                print j
                x, y, z, d = getdata(j)
                x2 = butter_bandpass_filter(x, 2*45, 2*51, Fs, order)
                y2 = butter_bandpass_filter(y, 2*45, 2*51, Fs, order)
                z2 = butter_bandpass_filter(z, 2*45, 2*51, Fs, order)
                xcorr2 = corr(x2, drive2)
                ycorr2 = corr(y2, drive2)
                zcorr2 = corr(z2, drive2)
                x = butter_bandpass_filter(x, 45, 51, Fs, order)
                y = butter_bandpass_filter(y, 45, 51, Fs, order)
                z = butter_bandpass_filter(z, 45, 51, Fs, order)
                xcorr = corr(x, drive1)
                xcorr = cali*xcorr
                ycorr = corr(y, drive1)
                zcorr = corr(z, drive1)
                X.append(xcorr)
                Y.append(ycorr)
                Z.append(zcorr)
                X2.append(xcorr2)
                Y2.append(ycorr2)
                Z2.append(zcorr2)

            L[i] = [X,Y,Z,X2,Y2,Z2]
            np.save(os.path.join(list_of_signal_folders[i], "measurement"), L[i])
        else:
            L[i] = np.load(os.path.join(list_of_signal_folders[i], "measurement.npy"))
    return [L, Cali]

def mean_and_err(L):
    ME = []
    
    for i in range(len(L)):
        ME2 = []
        for j in range(len(L[i])): # j counts x,y,z,x2,y2,z2
            s = L[i][j]
            h,b = np.histogram(s, bins = bins)

            bc = np.diff(b)/2 + b[:-1]

            p0 = [np.mean(s), np.std(s), len(s)/30.]
            try:
                af = [-1,-1]
                popt, pcov = curve_fit(gauss, bc, h, p0)
            except:
                af = [i,j]
                popt = p0
                pcov = (p0[1]**2)*np.ones([len(p0),len(p0)])/len(s)
            ME2.append([popt[0], np.sqrt(pcov[0][0])])
            
        ME.append(ME2)
    return [ME, af]


def Plots_corr(ME, tilt, Ca):
    if not len(ME) == len(tilt):
        return "len tilt is wrong"
    X = []
    Y = []
    Z = []
    Xerr = []
    Yerr = []
    Zerr = []
    X2 = []
    Y2 = []
    Z2 = []
    X2err = []
    Y2err = []
    Z2err = []
    for i in range(len(ME)):
        xf = ME[i][0][0]
        xferr = ME[i][0][1]

        yf = ME[i][1][0]
        yferr = ME[i][1][1]

        zf = ME[i][2][0]
        zferr = ME[i][2][1]

        xf2 = ME[i][3][0]
        xferr2 = ME[i][3][1]

        yf2 = ME[i][4][0]
        yferr2 = ME[i][4][1]

        zf2 = ME[i][5][0]
        zferr2 = ME[i][5][1]

        X.append(xf)
        Y.append(yf)
        Z.append(zf)

        Xerr.append(xferr)
        Yerr.append(yferr)
        Zerr.append(zferr)

        X2.append(xf2)
        Y2.append(yf2)
        Z2.append(zf2)

        X2err.append(xferr2)
        Y2err.append(yferr2)
        Z2err.append(zferr2)

    if fit_sub_plots:
        def func_fit(x, l, q, c):
            f = c + l*(x) + q*(x**2)
            return f
        
        datafit = [[X, Xerr], [Y, Yerr], [Z, Zerr], [X2, X2err], [Y2, Y2err], [Z2, Z2err]]
        fit_list = []
        fit_list_err = []
        for i in range(6):
            poptfit, pcovfit = curve_fit(func_fit, tilt, datafit[i][0], sigma = datafit[i][0])
            fit_list.append(poptfit)
            fit_list_err.append(pcovfit)
            
        space = np.linspace(tilt[0],tilt[-1], 1000)

    np.save(os.path.join(path_save,'2fxyz.txt'), (X2, Y2, Z2, tilt))
    fig = plt.figure()
    plt.subplot(3, 2, 2)
    ### plt.errorbar(tilt, X, yerr = Xerr, fmt = "b*", label="f")
    plt.errorbar(tilt, X2, yerr= X2err, xerr = tilterr, fmt = "ro", label="2f")
    if fit_sub_plots:
        plt.plot(space, func_fit(space, *fit_list[3]), "k--")
    plt.ylabel("xcorr")
    plt.legend()
    plt.grid()
    plt.subplot(3, 2, 4)
    # plt.errorbar(tilt, Y, yerr = Yerr, fmt = "b*", label="f")
    plt.errorbar(tilt, Y2, yerr= Y2err, xerr = tilterr, fmt = "ro", label="2f")
    if fit_sub_plots:
        plt.plot(space, func_fit(space, *fit_list[4]), "k--")
    plt.ylabel("ycorr")
    plt.legend()
    plt.grid()
    plt.subplot(3, 2, 6)
    # plt.errorbar(tilt, Z, yerr = Zerr, fmt = "b*", label="f")
    plt.errorbar(tilt, Z2, yerr= Z2err, xerr = tilterr, fmt = "ro", label="2f")
    if fit_sub_plots:
        plt.plot(space, func_fit(space, *fit_list[5]), "k--")
    plt.ylabel("zcorr")
    if voltage_scan:
        plt.xlabel("AC Voltage pp [x1000]")
    else:
        plt.xlabel("tilt motor #2 [rad]")
    #plt.xlabel("distance [arb]")
    plt.legend()
    plt.grid()
    plt.subplot(3, 2, 1)
    plt.errorbar(tilt, X, yerr = Xerr, xerr = tilterr, fmt = "b*", label="f")
    if fit_sub_plots:
        plt.plot(space, func_fit(space, *fit_list[0]), "k--")
    # plt.errorbar(tilt, X2, yerr= X2err,fmt = "ro", label="2f")
    plt.ylabel("xcorr <#e>")
    plt.legend()
    plt.grid()
    plt.subplot(3, 2, 3)
    plt.errorbar(tilt, Y, yerr = Yerr, xerr = tilterr, fmt = "b*", label="f")
    if fit_sub_plots:
        plt.plot(space, func_fit(space, *fit_list[1]), "k--")
    # plt.errorbar(tilt, Y2, yerr= Y2err,fmt = "ro", label="2f")
    plt.ylabel("ycorr")
    plt.legend()
    plt.grid()
    plt.subplot(3, 2, 5)
    plt.errorbar(tilt, Z, yerr = Zerr, xerr = tilterr, fmt = "b*", label="f")
    if fit_sub_plots:
        plt.plot(space, func_fit(space, *fit_list[2]), "k--")
    # plt.errorbar(tilt, Z2, yerr= Z2err,fmt = "ro", label="2f")
    plt.ylabel("zcorr")
    if voltage_scan:
        plt.xlabel("AC Voltage pp [x1000]")
    else:
        plt.xlabel("tilt motor #2 [rad]")
    plt.legend()
    plt.grid()
    plt.tight_layout(pad = 0)
    plt.savefig(os.path.join(path_save,'f_and_2f_corr_index100000.pdf'))

    if dist:
        fig = plt.figure()
        Xaxe = Ca
        plt.subplot(3, 2, 2)
        ### plt.errorbar(tilt, X, yerr = Xerr, fmt = "b*", label="f")
        plt.errorbar(Xaxe, X2, yerr= X2err, xerr = tilterr, fmt = "ro", label="2f")
        plt.ylabel("xcorr")
        plt.legend()
        plt.grid()
        plt.subplot(3, 2, 4)
        # plt.errorbar(tilt, Y, yerr = Yerr, fmt = "b*", label="f")
        plt.errorbar(Xaxe, Y2, yerr= Y2err, xerr = tilterr, fmt = "ro", label="2f")
        plt.ylabel("ycorr")
        plt.legend()
        plt.grid()
        plt.subplot(3, 2, 6)
        # plt.errorbar(tilt, Z, yerr = Zerr, fmt = "b*", label="f")
        plt.errorbar(Xaxe, Z2, yerr= Z2err, xerr = tilterr, fmt = "ro", label="2f")
        plt.ylabel("zcorr")
        # plt.xlabel("tilt motor #2 [rad]")
        plt.xlabel("1./Charge corr [Arb]")
        plt.legend()
        plt.grid()
        plt.subplot(3, 2, 1)
        plt.errorbar(Xaxe, X, yerr = Xerr, xerr = tilterr, fmt = "b*", label="f")
        # plt.errorbar(tilt, X2, yerr= X2err,fmt = "ro", label="2f")
        plt.ylabel("xcorr <#e>")
        plt.legend()
        plt.grid()
        plt.subplot(3, 2, 3)
        plt.errorbar(Xaxe, Y, yerr = Yerr, xerr = tilterr, fmt = "b*", label="f")
        # plt.errorbar(tilt, Y2, yerr= Y2err,fmt = "ro", label="2f")
        plt.ylabel("ycorr")
        plt.legend()
        plt.grid()
        plt.subplot(3, 2, 5)
        plt.errorbar(Xaxe, Z, yerr = Zerr, xerr = tilterr, fmt = "b*", label="f")
        # plt.errorbar(tilt, Z2, yerr= Z2err,fmt = "ro", label="2f")
        plt.ylabel("zcorr")
        plt.xlabel("1./Charge corr [Arb]")
        plt.legend()
        plt.grid()
        plt.tight_layout(pad = 0)
        plt.savefig(os.path.join(path_save,'f_and_2f_corr_xcharge_index100000.pdf'))


if len(list_of_signal_folders) == len(tilt):
    print "continue"

L, Cali = corr_in_folders(list_of_signal_folders, list_of_charge_folders)

ME, af = mean_and_err(L)

Plots_corr(ME, tilt, 1.0/np.array(Cali))

if af[0] != -1:
    print "some of the gauss fit failed, the last being [i,j] = ", af

if not dont_calibrate:
    plt.figure()
    plt.plot(tilt, Cali, "ro")
    plt.plot(tilt, 1.0/np.array(Cali), "bo")
    plt.xlabel("distance [Arb]")
    plt.ylabel("Correlation [Arb]")
    plt.savefig(os.path.join(path_save,'1p_corr.pdf'))
    plt.grid()
else:
    print "!!!!!!!!!!!!!!!!!!!!!!!!!!CAREFULL CALIBRATION IS OFF!!!!!!!!!!!!!!!!!!!!!"
    print "!!!!!!!!!!!!!!!!!!!!!!!!!!CAREFULL CALIBRATION IS OFF!!!!!!!!!!!!!!!!!!!!!"
    print "!!!!!!!!!!!!!!!!!!!!!!!!!!CAREFULL CALIBRATION IS OFF!!!!!!!!!!!!!!!!!!!!!"
    print "!!!!!!!!!!!!!!!!!!!!!!!!!!CAREFULL CALIBRATION IS OFF!!!!!!!!!!!!!!!!!!!!!"
    print "!!!!!!!!!!!!!!!!!!!!!!!!!!CAREFULL CALIBRATION IS OFF!!!!!!!!!!!!!!!!!!!!!"
    print "!!!!!!!!!!!!!!!!!!!!!!!!!!CAREFULL CALIBRATION IS OFF!!!!!!!!!!!!!!!!!!!!!"

plt.show()










