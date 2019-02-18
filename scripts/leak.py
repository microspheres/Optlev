import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt


path = r"C:\data\leak\deflection"

filelist = ["0.6W.h5", "0.8W.h5" ,"1.0W.h5", "1.2W.h5", "1.4W.h5", "1.6W.h5" ,"1.8W.h5" ,"2W.h5"]

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**17

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
                pid = dset.attrs['PID']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT) 
	ypsd, freqs = matplotlib.mlab.psd(dat[:, bu.yi]-numpy.mean(dat[:, bu.yi]), Fs = Fs, NFFT = NFFT)
        zpsd, freqs = matplotlib.mlab.psd(dat[:, bu.zi]-numpy.mean(dat[:, bu.zi]), Fs = Fs, NFFT = NFFT)
        # Ddrive = dat[:, bu.drive]*np.gradient(dat[:,bu.drive])
        # DdrivePSD, freqs =  matplotlib.mlab.psd(Ddrive-numpy.mean(Ddrive), Fs = Fs, NFFT = NFFT))
        print pid
	return [freqs, xpsd, ypsd, dat, zpsd]



p = 38

A = []

Power = []

plt.figure()
for i in filelist:
    data = getdata(os.path.join(path, i))
    a = np.sum(data[1][p-1:p+1])
    a = np.sqrt(a)
    df = data[0][1]
    a = a*df
    power = float(str(i).split("W.")[0])
    A.append(a)
    Power.append(power)
    plt.loglog(data[0], np.sqrt(data[1]))
    plt.plot(data[0][p], np.sqrt(data[1][p]), "rx")




def fit(x, a, b, c):
    return a*(x**2) + b*x + c

popt, pcov = opt.curve_fit(fit ,Power, A)

PP = np.arange(0.6, 2, 0.01)

plt.figure()
plt.plot(Power, A, "ro")
plt.plot(PP, fit(PP, *popt))
plt.ylabel("X sensor [V]")
plt.xlabel("Laser Power [W]")

print popt
print pcov

a = "a = " + str(popt[0]) + "+-" +  str(np.sqrt(pcov[0][0]))
b = "b = " + str(popt[1]) + "+-" +  str(np.sqrt(pcov[1][1])) 

print a
print b

plt.show()

