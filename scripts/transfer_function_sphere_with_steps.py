import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import os, re, time, glob
from scipy.signal import butter, lfilter, filtfilt
from scipy import signal
import bead_util as bu
import glob


def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

start_index = 0

# freq_list = np.logspace(1.5, 3.5, 20)
freq_list = [8.,32., 40., 48., 56., 64., 72., 80.,96., 128., 168., 200., 208., 256., 512., 1024., 2048.]

path_charge = r"C:\data\20171004\bead9_15um_QWP_NS\calibration1e\5"

path_signal = r"C:\data\20171004\bead9_15um_QWP_NS\meas\steps\comb3"

endfile = -1

startfile = 0

start_index = 0

file_list_signal = glob.glob(path_signal+"\*.h5")
file_list_charge = glob.glob(path_charge+"\*.h5")

file_list_signal = list_file_time_order(file_list_signal)

file_list_signal = file_list_signal[startfile:endfile]


order = 3

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

p = bu.drive
def getdata_x_d(fname):
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		Fs = dset.attrs['Fsamp']
		dat = dat * 10./(2**15 - 1)
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )
        
	x = dat[start_index:, 0]-numpy.mean(dat[start_index:, 0])
	driveN = ((dat[start_index:, p] - numpy.mean(dat[start_index:, p])))/np.max((dat[start_index:, p] - numpy.mean(dat[start_index:, p])))
	
	return [x, driveN]


def transf(list_charge, list_signal, freq_list):
    A = [] # tranf for each freq
    drive = getdata_x_d(list_charge[0])[1]
    counter = 0.
    for i in range(len(freq_list)):
        d = butter_bandpass_filter(drive, freq_list[i]-3, freq_list[i]+3, 10000, order)
        dfft = np.fft.rfft(d)
        arg = np.argmax(dfft)
        a = 0
        for j in range(len(list_signal)):
            x = getdata_x_d(list_signal[j])[0]
            xfft = np.fft.rfft(x)
            t = xfft/dfft
            a = a + t[arg]
            counter = counter + 1
            print counter*len(list_charge)/(len(list_signal))
        a = a/len(list_signal)
        A.append(a)
    return A


#####################################################################
#####################################################################
# because of the steps

def organize_DC_pos(list):
    file_list_new = []
    for i in range(len(list)):
        if float(re.findall("-?\d+mVDC",list[i])[0][:-4]) > 0:
            file_list_new.append(list[i])
    return file_list_new

def organize_DC_neg(list):
    file_list_new = []
    for i in range(len(list)):
        if float(re.findall("-?\d+mVDC",list[i])[0][:-4]) < 0:
            file_list_new.append(list[i])
    return file_list_new

file_list_pos = organize_DC_pos(list_file_time_order(file_list_signal))
file_list_neg = organize_DC_neg(list_file_time_order(file_list_signal))

l1 = len(file_list_pos)
l2 = len(file_list_neg)
lmin = np.min([l1,l2])

file_list_pos = file_list_pos[:lmin]
file_list_neg = file_list_neg[:lmin]



Apos = np.array(transf(file_list_charge, file_list_pos, freq_list))
Aneg = np.array(transf(file_list_charge, file_list_neg, freq_list))


A = 0.5*(Apos + Aneg)

plt.figure()
plt.loglog(freq_list,abs(np.real(A)), marker = "o", linestyle = "--", color = "r")
plt.ylabel("transf function arb units")
plt.xlabel("frequency [Hz]")
plt.grid()
plt.figure()
plt.semilogx(freq_list,np.imag(np.log(A)), marker = "o", linestyle = "--", color = "r")
plt.ylabel("phase")
plt.xlabel("frequency [Hz]")
plt.grid()
plt.show()

        
