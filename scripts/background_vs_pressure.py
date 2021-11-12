import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob


path = r"C:\data\20190619\test_background2"

NFFT = 2**19

F1 = 15.2
F2 = 15.3

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
                Press = dset.attrs['pressures'][0]
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT) 
	ypsd, freqs = matplotlib.mlab.psd(dat[:, bu.yi]-numpy.mean(dat[:, bu.yi]), Fs = Fs, NFFT = NFFT)
        zpsd, freqs = matplotlib.mlab.psd(dat[:, bu.zi]-numpy.mean(dat[:, bu.zi]), Fs = Fs, NFFT = NFFT)

	return [freqs, xpsd, ypsd, dat, zpsd, Press]


file_list = glob.glob(path+"\*.h5")

def get_power_file(f):
        a = getdata(f)
        freq = a[0]

        f1 = np.where((freq) > F1)[-1][0]
        f2 = np.where((freq) < F2)[-1][-1]

        x = np.sqrt(np.sum(a[1][f1:f2]))

        x_b = np.sqrt(np.sum(a[1][f2:(2*f2-f1)]))
        
        y = np.sqrt(np.sum(a[2][f1:f2]))
        
        y_b = np.sqrt(np.sum(a[2][f2:(2*f2-f1)]))
        
        z = np.sqrt(np.sum(a[4][f1:f2]))
        
        z_b = np.sqrt(np.sum(a[4][f2:(2*f2-f1)]))

        press = a[5]

        return [press, x, y, z, x_b, y_b, z_b]

def get_power(file_list):
    P = []
    X = []
    Y = []
    Z = []
    Xb = []
    Yb = []
    Zb = []
    for i in file_list:
        a = get_power_file(i)
        P.append(a[0])
        X.append(a[1])
        Y.append(a[2])
        Z.append(a[3])
        Xb.append(a[4])
        Yb.append(a[5])
        Zb.append(a[6])

    return [P, X, Y, Z, Xb, Yb, Zb]

V = get_power(file_list)

plt.figure()
plt.subplot(3, 1, 1)
plt.loglog(V[0],V[1], "ro", label = "X")
plt.loglog(V[0],V[4], "rx", label = "X_noise")
plt.ylabel("Voltage")
plt.legend()
plt.grid()
plt.subplot(3, 1, 2)
plt.loglog(V[0],V[2], "bo", label = "Y")
plt.loglog(V[0],V[5], "bx", label = "Y_noise")
plt.ylabel("Voltage")
plt.legend()
plt.grid()
plt.subplot(3, 1, 3)
plt.loglog(V[0],V[3], "go",label = "Z")
plt.loglog(V[0],V[6], "gx",label = "Z_noise")
plt.ylabel("Voltage")
plt.legend()
plt.grid()

plt.xlabel("Pressure [mbar]")

plt.tight_layout(pad = 0)
plt.show()

        
