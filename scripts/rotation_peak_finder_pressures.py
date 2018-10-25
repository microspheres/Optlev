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


path1 = r"C:\data\201712018\bead3_um_QWP_NS_VAT\damp1_inverse_to_110deg"


file_list1 = glob.glob(path1+"\*.h5")
file_list1 = list_file_time_order(file_list1)
file_list1 = file_list1



Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**16


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
                time = dset.attrs['Time']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2] )

        xpsd_old, freqs = matplotlib.mlab.psd(dat[:, 0]-numpy.mean(dat[:, 0]), Fs = Fs, NFFT = NFFT)
        # Ddrive = dat[:, bu.drive]*np.gradient(dat[:,bu.drive])
        # DdrivePSD, freqs =  matplotlib.mlab.psd(Ddrive-numpy.mean(Ddrive), Fs = Fs, NFFT = NFFT)

        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,bu.zi])**2
	return [freqs, 0, 0, 0, 0, xpsd_old, Press, time]


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
    # angle = []
    P = []
    aux_press = str("mbar")
    press = str(getdata(filelist[0])[6]) + aux_press
    T = []

    for i in filelist:
        gd = getdata(i)
        pol_sens = gd[5][argcut:argcutabove]
        arg = np.argmax(pol_sens)
        rot = freq[arg+argcut]
        # ang = i[i.rfind('_')+1:i.rfind('deg.h5')]
        if np.max(pol_sens) > cut:
            rot = float(rot)
        else:
            rot = float(0)
        rotation.append(rot)
        # angle.append(float(ang))
        P.append(gd[6])
        T.append(gd[7])
    return [rotation, P, np.array(T)-T[0]]



c1 = finder(file_list1, path1, 30000., 2200000, 1.e-11)


# fitting

# def step(x,cut):
#     f = 0.0
#     if x > cut:
#         f = 1.
#     return f

# def func(x, k1, k2, off):
#     a = np.pi/180.
#     f = np.abs(k1*2.0*np.sin(a*(2.0*(k2*x + off))))
#     return f

# p0 = [10000., 0.95, 10.]
# popt, pcov = curve_fit(func, c1[1][0:40], c1[0][0:40], p0 = np.array(p0))

# print popt

# angles =  np.linspace(25, 90, 3000)




plt.figure()
plt.plot(c1[1], c1[0], "bo", label = "QWP = 50 deg")


# plt.plot(angles, func(angles, *popt), "k-", label = "fit")


plt.legend(loc="upper right", frameon = False)
plt.ylabel("Rotation [Hz]", fontsize=13)
plt.xlabel("Pressure [mbar]", fontsize=13)
plt.grid()
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout(pad = 0)

plt.figure()
plt.scatter(c1[2], c1[0], s=7, c=np.log10(c1[1]), label = "QWP = 50 deg")
plt.colorbar()
plt.show()
