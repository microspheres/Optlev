import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
from scipy.signal import butter, lfilter, filtfilt
import scipy.optimize as opt


single_channel = True
VEOM_h5 = True
measuring_with_xyz = False

scope = False # gets the correct FS

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

#path = r"C:\data\20190211\15um\2\rot4"
path = r"C:\data\20190725\15um_SiO2\4\pressures\daq"

#path = r"C:\data\20190725\15um_SiO2\4\pressures\daq\KEYSIGHT_362337Hz"

file_list = glob.glob(path+"\*.h5")

file_list = list_file_time_order(file_list)

NFFT = 2**20

if single_channel:
    a = 0
else:
    a = bu.xi_old

if measuring_with_xyz:
    a = -1

    
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
                Press = dset.attrs['pressures'][0]
                Volt = 0.0
                if VEOM_h5:
                    Volt = dset.attrs['EOM_voltage']
                if scope:
                    Fs = dset.attrs['FS_scope']
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                Time = dset.attrs["Time"]
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2] )

        xpsd_old, freqs = matplotlib.mlab.psd(dat[:, a]-numpy.mean(dat[:, a]), Fs = Fs, NFFT = NFFT)

        rot = dat[:, a]-numpy.mean(dat[:, a])
        
	return [freqs, 0, 0, 0, rot, xpsd_old, Press, Volt, Time, Fs]


order = 2
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

def funcfit(t, f0, phase, A):
    f = A*np.sin(phase + 2.0*np.pi*f0*t)
    return f



data = getdata(file_list[0])

# x = butter_bandpass_filter(data[4], 280000., 370000., data[9], order)
# x = butter_bandpass_filter(data[4], 118500., 120500., data[9], order)
x = butter_bandpass_filter(data[4], 363000 - 100000., 363000 + 100000., data[9], order)


# xfft, f = matplotlib.mlab.psd(data[4], Fs = data[9], NFFT = NFFT)
# xfftf, f1 = matplotlib.mlab.psd(x[len(x)/2:len(x)/2 + 1000], Fs = data[9], NFFT = NFFT)

# plt.figure()
# plt.loglog(f, xfft)
# plt.loglog(f1, xfftf)
# plt.figure()
# plt.plot(x)
# plt.show()


# time = np.arange(len(x))/data[9]

Fs = data[9]

II = (8./15.)*np.pi*(1600.)*((7.5e-6)**5)

def freq_fit(x, Fs, time_fit):
    
    time = np.arange(len(x))/Fs
    div = (len(x)/Fs)/time_fit
    a = 0
    b = int(len(x)/div)
    F = []
    dF = []
    T = []
    number = 0
    Resi = []

    f0 = 362337.
    p0 = np.array([f0, 0.04, 0.17])
    bounds = ([f0 - 0.5*f0, 0.0, 0.0], [f0 + 0.5*f0, 2.*np.pi, 5.])
    for i in range(int(div)):

        try:
            p, c = opt.curve_fit(funcfit, time[0:0+b], x[a:a+b], p0 = p0)
            # print c
            p0 = p
            aaa = True
        except:
            aaa = False
            print "fit is failing"

        number = number + 1
        t = a/Fs
 
        if number/div < 1:
            resi = np.sqrt(np.abs(np.sum((x[a:a+b]**2 - funcfit(time[0:0+b], *p)**2))))
        print "doing:", number/div

        if aaa:
            F.append(p[0])
            dF.append(c[0][0])
            T.append(t)
            Resi.append(resi)
            if (number % 16409 == 0):
                plt.figure()
                plt.plot(time[a:a+b], x[a:a+b])
                plt.plot(time[a:a+b], funcfit(time[0:0+b], *p))
        a = a + b

    t2 = len(x)/Fs
    Fs2 = len(F)/t2
    print len(F)
    psd, freq = matplotlib.mlab.psd(F - np.mean(F), Fs = Fs2, NFFT = 2**13)

    return [F, dF, T, freq, psd, Resi]

filename = os.path.join(path, "200us_block")

f = freq_fit(x, Fs, 0.0002)

np.save(filename, f)

plt.figure()
plt.errorbar(f[2], f[0], np.sqrt(f[1]), fmt = ".-")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
plt.grid()
plt.legend()
plt.figure()
#plt.loglog(f[3], 2.*np.pi*II*np.sqrt(f[4]))
plt.loglog(f[3], f[3]*2.*np.pi*II*np.sqrt(f[4]))
plt.ylabel("Nm / sqrt Hz")
plt.xlabel("Frequency [Hz]")
plt.grid()
plt.legend()
plt.figure()
plt.loglog(f[3], np.sqrt(f[4]))
plt.ylabel("$\Delta f$ / sqrt Hz")
plt.xlabel("Frequency [Hz]")
plt.grid()
plt.legend()
plt.figure()
#plt.loglog(f[3], 2.*np.pi*np.sqrt(f[4]))
plt.loglog(f[3], f[3]*2.*np.pi*np.sqrt(f[4]))
plt.ylabel("angular acc / sqrt Hz")
plt.xlabel("Frequency [Hz]")
plt.grid()
plt.legend()

# plt.figure()
# plt.plot(f[2], f[5])
# plt.ylabel("Residuals")
# plt.grid()
# plt.legend()
    


plt.show()


# single_channel = True
# VEOM_h5 = True
# measuring_with_xyz = False

# scope = True # gets the correct FS

# def list_file_time_order(filelist):
#     filelist.sort(key=os.path.getmtime)
#     return filelist


# path = r"C:\data\20190211\15um\2\rot4"

# file_list = glob.glob(path+"\*.h5")

# file_list = list_file_time_order(file_list)

# NFFT = 2**17

# if single_channel:
#     a = 0
# else:
#     a = bu.xi_old

# if measuring_with_xyz:
#     a = -1
