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

electron = 1.60E-19

d = 0.0008

Vpp = 160.0

conversion41 = 0.102098631501

conversion82 = -6.74639623327e-05

Vmeasurement_pp = 200.0*8.5

Nucleons = 1.1E15

path_charge = r"C:\data\20170717\bead15_15um_QWP\steps\calibration_charge"

path_signal = r"C:\data\20170717\bead15_15um_QWP\steps\measurement_2"

path_noise = r"C:\data\20170717\bead15_15um_QWP\steps\calibration_charge"

p = bu.drive

fdrive = 39.

Fs = 10000

li = 30.

ls = 200.

butterp = 1



file_list_noise = glob.glob(path_noise+"\*.h5")
file_list_signal = glob.glob(path_signal+"\*.h5")
file_list_charge = glob.glob(path_charge+"\*.h5")

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
        
	x = dat[:, 0]-numpy.mean(dat[:, 0])
        x =  butter_bandpass_filter(x, li, ls, Fs, butterp)
	driveN = ((dat[:, p] - numpy.mean(dat[:, p])))/np.max((dat[:, p] - numpy.mean(dat[:, p])))

	driveNf = butter_bandpass_filter(driveN, li, ls, Fs, butterp)/np.max(butter_bandpass_filter(driveN, li, ls, Fs, butterp))
    
	drive2W = (driveNf*driveNf - np.mean(driveNf*driveNf))/np.max(driveNf*driveNf - np.mean(driveNf*driveNf))
	
	return [x, driveN, drive2W]


def getdata_noise(fname):
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		Fs = dset.attrs['Fsamp']
		dat = dat * 10./(2**15 - 1)
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )
        
	x = dat[:, 0]-numpy.mean(dat[:, 0])

	return [x]


def getphase(fname, driveN, x, Fs):

    xdat = np.append(x, np.zeros( int(Fs/fdrive) ))
    corr2 = np.correlate(xdat, driveN)
    maxv = np.armax(corr2) 
	
    return maxv


def Jnoise(noise_file, maxv):
    J = []
    zero = np.zeros(maxv)
    noise_aux = []
    fftnoise = []
    a = 0
    for i in noise_file:
        noise_aux = getdata_noise(i)
        shift_N = np.append(noise_aux, zero)
        fftnoise = np.fft.rfft(shift_N)
        j = np.abs(np.conjugate(fftnoise)*fftnoise)
        J = j if J == [] else J + np.array(j)
        a += 1.
        print "PSDnoise", a/len(noise_file)
    return (J/len(noise_file))**0


def corr_aux(drive2WN, driveN, x, Jnoise, maxv):
    zero = np.zeros(maxv)
    shift_x = np.append(x, zero)
    shift_d = np.append(zero, driveN)
    shift_d2W = np.append(zero, drive2WN)

    fftx = np.fft.rfft(shift_x)
    fftd = np.fft.rfft(shift_d)
    fftd2W = np.fft.rfft(shift_d2W)

    jx = Jnoise

    Fi = np.argmax(fftd2W) - 5
    Fs = np.argmax(fftd2W) + 5

    boundi = 1500
    bounds = 7500

    # boundi = np.argmax(fftd) - 10
    # bounds = np.argmax(fftd) + 10


    corr = np.sum(np.conjugate(fftd[boundi:bounds])*fftx[boundi:bounds]/jx[boundi:bounds])/np.sum(np.conjugate(fftd[boundi:bounds])*fftd[boundi:bounds]/jx[boundi:bounds])
    corr = corr

    corr2W = np.sum(np.conjugate(fftd2W[Fi:Fs])*fftx[Fi:Fs]/jx[Fi:Fs])/np.sum(np.conjugate(fftd2W[Fi:Fs])*fftd2W[Fi:Fs]/jx[Fi:Fs])
    corr2W = corr2W

    return [corr, corr2W]





def get_drive(list):
    x, d, d2 = getdata_x_d(list[0])
    return [d, d2]




def do_corr(file_list_signal, maxv, Jnoise):
    corr = np.zeros(len(file_list_signal), complex)
    corr2W = np.zeros(len(file_list_signal), complex)
    a = 0
    for i in range(len(file_list_signal)):
        x, d, d2 = getdata_x_d(file_list_signal[i])
        c1, c2 = corr_aux(d2, d, x, Jnoise, maxv)
        corr[i] = (c1)
        corr2W[i] = (c2)
        a += 1.
        print "corr", a/len(file_list_signal)
    return [corr, corr2W]





def do_corr_trigger(file_list_signal, maxv, Jnoise, d, d2):
    corr = np.zeros(len(file_list_signal), complex)
    corr2W = np.zeros(len(file_list_signal), complex)
    a = 0
    for i in range(len(file_list_signal)):
        x, emp1, emp2 = getdata_x_d(file_list_signal[i])
        c1, c2 = corr_aux(d2, d, x, Jnoise, maxv)
        corr[i] = (c1)
        corr2W[i] = (c2)
        a += 1.
        print "corr", a/len(file_list_signal)
    return [corr, corr2W]

def find_voltagesDC(list):
    aux = np.zeros(len(list))
    for i in range(len(list)):
        aux[i] = float(re.findall("-?\d+VDC",list[i])[0][:-3])
    return aux

def find_voltagesAC(list):
    aux = np.zeros(len(list))
    for i in range(len(list)):
        aux[i] = float(re.findall("-?\d+mV",list[i])[0][:-2])
    return aux

def get_all_V(Vlist):
    x = list(set(Vlist))
    return x

def do_corr_for_VCD(file_list, maxv, Jnoise, d, d2, VDC):
    files_vdc = []
    for i in range(len(file_list)):
        if VDC == float(re.findall("-?\d+VDC",file_list[i])[0][:-3]):
            files_vdc.append(file_list[i])
    corr, corr2 = do_corr_trigger(files_vdc, maxv, Jnoise, d, d2)
    Vpp_AC_measurement = find_voltagesAC(files_vdc)[0]
    corr = corr*(Vpp/(0.200*Vpp_AC_measurement))*(1.0/conversion41)/Nucleons
    return np.real(corr)


def corr_all_VDC(file_list, maxv, Jnoise, d, d2, VDClist):
    corr = []
    for i in range(len(VDClist)):
        corra = np.mean(do_corr_for_VCD(file_list, maxv, Jnoise, d, d2, VDClist[i]))
        corr.append(corra)
    return corr

def SUM(corrlist,vdclist):
    corr = []
    for i in range(len(vdclist)):
        a = (corrlist[i] + corrlist[7-i])/2.
        corr.append(a)
    return corr
        




VDClist = get_all_V(find_voltagesDC(file_list_signal))

Jx = Jnoise(file_list_noise, 0)

drive_t, drive2_t = get_drive(file_list_charge)

corrlist = corr_all_VDC(file_list_signal, 0, Jx, drive_t, drive2_t, VDClist) 

corr = SUM(corrlist, VDClist)

print corr
print VDClist[0:3]

plt.figure()
plt.plot(VDClist[0:3], corr)
plt.show()
