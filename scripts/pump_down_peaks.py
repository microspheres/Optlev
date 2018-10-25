import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
	 
# plot peaks during pump down

path_charge = r"C:\data\20171004\bead9_15um_QWP_NS\calibration1e\1"
file_list_charge = glob.glob(path_charge+"\*.h5")

path_psd = r"C:\data\20171004\bead9_15um_QWP_NS\pressures"
file_list_psd = glob.glob(path_psd+"\*.h5")

v_calibration = 0.1 # vpp in the daq
v_calibration = v_calibration/2.0 # now v is in amplitude

NFFT = 2**15

gain_factor = 20. # plots measured at gain 5 and calibration gain 100

startfile = 0
endfile = 150

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

file_list_psd = list_file_time_order(file_list_psd)

file_list_psd = file_list_psd[startfile:endfile]

counter = 10
ratio = 30
nlist = []
for i in range(len(file_list_psd)):
    if counter + ratio*i >= len(file_list_psd):
        file_list_psd = nlist
    else:
        a = file_list_psd[counter + ratio*i]
        nlist.append(a)


print file_list_psd

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
		pressure = dset.attrs['temps'][0]
		
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

def v_to_newton(file_list):
    distance = 0.001 #m
    v = 200.0*v_calibration # volts
    E = 1.0*v/distance
    charge = 1.602*10**(-19) # SI units
    force = charge*E
    conversion = force/np.sqrt(Voltsquare_at_peak(file_list))
    return conversion
 

conversion_N = v_to_newton(file_list_charge)*gain_factor


plt.figure()
for file in file_list_psd:
    psd = getdata(file)
    plt.loglog(psd[0], conversion_N*np.sqrt(psd[1]), label = str(psd[5])+" mbar")

plt.legend()
plt.xlabel("frequency [Hz]")
plt.ylabel("Force [N]")
plt.grid()
plt.show()
