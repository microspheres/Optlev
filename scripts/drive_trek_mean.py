import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os, re
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
from scipy.optimize import curve_fit

path_charge = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\several_distances\back3_and_M2tilt\M2tilt_0\calibration1p"
file_list_charge = glob.glob(path_charge+"\*.h5")

p = bu.drive


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

	d = dat[:, p]
	return d

D  = []
D1 = []
for i in file_list_charge:
    d = getdata(i)
    s = np.mean(d)
    D.append(s)
    D1.append(d)

def sine(x, A, ph, w, c):
    return A*np.sin(w*x + ph)+c

p0 = np.array([1., 1e-5, 1/33., 1e-6])
X = np.linspace(0, len(D1[0]), len(D1[0]))

DF = D1[0] + D1[1]

popt, pcov = curve_fit(sine, X[0:800], DF[0:800], p0 = p0)

print D
print popt
print "ratio", popt[3]/popt[0]
print "V_off at max V", 200*popt[3]/2

print (3.0e-17)*(4)*(popt[3]/popt[0])

plt.figure()
plt.plot(X[0:800],DF[0:800])
plt.plot(X[0:800],sine(X[0:800], *popt))
plt.show()
