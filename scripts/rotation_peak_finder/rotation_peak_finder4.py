import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import bead_util as bu
import glob
from scipy.optimize import curve_fit


def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist


path1 = r"C:\data\20190108\15um\rotation"
path2 = r"C:\data\20190108\15um\rotation"



file_list1 = glob.glob(path1+"\*.h5")
file_list1 = list_file_time_order(file_list1)
file_list1 = file_list1

file_list2 = glob.glob(path2+"\*.h5")
file_list2 = list_file_time_order(file_list2)
file_list2 = file_list2


Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**12


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
                Press = dset.attrs['temps'][0]
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT) 
	ypsd, freqs = matplotlib.mlab.psd(dat[:, bu.yi]-numpy.mean(dat[:, bu.yi]), Fs = Fs, NFFT = NFFT)
        zpsd, freqs = matplotlib.mlab.psd(dat[:, bu.zi]-numpy.mean(dat[:, bu.zi]), Fs = Fs, NFFT = NFFT)
        xpsd_old, freqs = matplotlib.mlab.psd(dat[:, bu.xi_old]-numpy.mean(dat[:, bu.xi_old]), Fs = Fs, NFFT = NFFT)
        # Ddrive = dat[:, bu.drive]*np.gradient(dat[:,bu.drive])
        # DdrivePSD, freqs =  matplotlib.mlab.psd(Ddrive-numpy.mean(Ddrive), Fs = Fs, NFFT = NFFT)


	norm = numpy.median(dat[:, bu.zi])
        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,bu.zi])**2
	return [freqs, xpsd, ypsd, dat, zpsd, xpsd_old, Press]


def return_arg(list, value):
    aux = 1.0*np.array(list) - 1.0*value
    aux = np.abs(aux)
    arg = np.argmin(aux)
    return arg


def finder(filelist, path, freq_cut, freq_cutabove, cut):
    freq = getdata(filelist[0])[0]
    argcut = return_arg(freq, freq_cut)
    argcutabove = return_arg(freq, freq_cutabove)

    rotation = []
    angle = []
    P = []
    aux_press = str("mbar")
    press = str(getdata(filelist[0])[6]) + aux_press


    for i in filelist:
        pol_sens = getdata(i)[5][argcut:argcutabove]
        arg = np.argmax(pol_sens)
        rot = freq[arg+argcut]
        ang = i[i.rfind('_')+1:i.rfind('deg.h5')]
        if np.max(pol_sens) > cut:
            rot = float(rot)
        else:
            rot = float(0)
        rotation.append(rot)
        angle.append(float(ang))
    return [rotation, angle, press]



c1 = finder(file_list1, path1, 3800., 20000., 6.e-11)
c2 = finder(file_list2, path2, 2800., 25000., 6.e-11)


# fitting

def step(x,cut):
    f = 0.0
    if x > cut:
        f = 1.
    return f

def func(x, k1, k2, off):
    a = np.pi/180.
    f = np.abs(k1*2.0*np.sin(a*(2.0*(k2*x + off))))
    return f

p0 = [10000., 0.95, 10.]
popt, pcov = curve_fit(func, c2[1][0:40], c2[0][0:40], p0 = np.array(p0))

print popt

angles =  np.linspace(25, 90, 3000)




plt.figure()
plt.plot(c1[1], c1[0], "bo", label = c1[2])
plt.plot(c2[1], c2[0], "ro", label = c2[2])


plt.plot(angles, func(angles, *popt), "k-", label = "fit")


plt.legend(loc="upper right", frameon = False)
plt.ylabel("Rotation [Hz]")
plt.xlabel("Angle [degrees]")
plt.grid()
plt.show()
