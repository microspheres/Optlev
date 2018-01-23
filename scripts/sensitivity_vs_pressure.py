import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

	 
path_charge = r"C:\data\20171031\bead1_15um_QWP_NS\calibration1e\1"
file_list_charge = glob.glob(path_charge+"\*.h5")

path_psd = r"C:\data\20171031\bead1_15um_QWP_NS\pump_up"
file_list_psd = glob.glob(path_psd+"\*.h5")

mass = (2.66*10**-12) # in kg

Number_of_e = (8.00*10**14)

v_calibration = 0.5 # vpp in the daq
v_calibration = v_calibration/2.0 # now v is in amplitude

distance = 0.0021 #m

NFFT = 2**17

startfile = 0
endfile = -1

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

file_list_psd = list_file_time_order(file_list_psd)

file_list_psd = file_list_psd[startfile:endfile]


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
                pressure = dset.attrs['temps']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT)
        drivepsd, freqs = matplotlib.mlab.psd(dat[:, bu.drive]-numpy.mean(dat[:, bu.drive]), Fs = Fs, NFFT = NFFT)
        aux = np.argmax(drivepsd)
        freq_drive = freqs[aux]
	return [freqs, xpsd, drivepsd, freq_drive, aux, pressure]

def unite_psd(file_list): # for charge calibration
    freqs = np.array(getdata(file_list[0])[0])
    X = np.zeros(len(freqs))
    for file in file_list:
       a = getdata(file)
       X += np.array(a[1])
    return [freqs, X/len(file_list)]

def Voltsquare_at_peak(file_list): # area of the peak
    a = 2
    freq, xpsd = unite_psd(file_list)
    peak = getdata(file_list[0])[4]
    dfreq = freq[peak] - freq[peak - 1]
    v2 = np.sum(xpsd[peak - a:peak + a])*dfreq
    return v2

def v_to_g(file_list):
    v = 200.0*v_calibration # volts
    E = 1.0*v/distance
    charge = 1.602*10**(-19) # SI units
    force = charge*E
    a = force/mass
    conversion = (a/np.sqrt(Voltsquare_at_peak(file_list)))/9.8
    return conversion

def index(array, number1, number2):
    a = np.ones(len(array))*number1
    b = np.abs(array - a)
    c = np.where(b > 0)
    d = np.where(b < number2 - number1)
    e = np.intersect1d(c,d)
    return e

def plot_pressure_vs_g(file_list_psd, file_list_charge): # check that the sensor gain for calibration is the same for the measurement!
    # A = unite_psd(file_list_psd)
    c = v_to_g(file_list_charge)
    psd = []
    pre = []
    for i in file_list_psd:
        aux = getdata(i)
        f = np.array(aux[0])
        x = np.array(aux[1])
        j = index(f, 47., 48.)
        x1 = np.mean(x[j[0]:j[-1]])
        x1 = c*np.sqrt(x1)
        psd.append(x1)
        pre.append(aux[5][0])
    plt.figure()
    plt.plot(pre,1e6*np.array(psd), "ro")
    plt.ylabel('acceleration [$\mu$g]/$\sqrt{Hz}$')
    plt.xlabel('Pressure[mbar]')
    plt.plot()
    plt.grid()
    return



plot_pressure_vs_g(file_list_psd, file_list_charge)
plt.show()
