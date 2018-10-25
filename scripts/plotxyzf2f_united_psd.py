import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os, re
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob


path_charge = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\electrode_dist_fartherN6\calibration_1p"
file_list_charge = glob.glob(path_charge+"\*.h5")

path_save = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\electrode_dist_fartherN6"


# list_of_psd_folders = [r"C:\data\20180323\bead3_SiO2_15um_POL_NS\electrode_dist_fartherN5\meas1_tilt_0", r"C:\data\20180323\bead3_SiO2_15um_POL_NS\electrode_dist_fartherN5\meas1_M2_tilt_1000", r"C:\data\20180323\bead3_SiO2_15um_POL_NS\electrode_dist_fartherN5\meas1_M2_tilt_2000", r"C:\data\20180323\bead3_SiO2_15um_POL_NS\electrode_dist_fartherN5\meas1_M2_tilt_3000", r"C:\data\20180323\bead3_SiO2_15um_POL_NS\electrode_dist_fartherN5\meas1_M2_tilt_5000"]

# tilt = [0, 1000, 2000, 3000, 5000]

list_of_psd_folders = [r"C:\data\20180323\bead3_SiO2_15um_POL_NS\electrode_dist_fartherN6\meas1_M2_tilt_0", r"C:\data\20180323\bead3_SiO2_15um_POL_NS\electrode_dist_fartherN6\meas2_M2_tilt_5000"]

tilt = [0, 5000]


NFFT = 2**19

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

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

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT)
        ypsd, freqs = matplotlib.mlab.psd(dat[:, bu.yi]-numpy.mean(dat[:, bu.yi]), Fs = Fs, NFFT = NFFT)
        zpsd, freqs = matplotlib.mlab.psd(dat[:, bu.zi]-numpy.mean(dat[:, bu.zi]), Fs = Fs, NFFT = NFFT)
        drivepsd, freqs = matplotlib.mlab.psd(dat[:, bu.drive]-numpy.mean(dat[:, bu.drive]), Fs = Fs, NFFT = NFFT)
        aux = np.argmax(drivepsd)
        freq_drive = freqs[aux]
	return [freqs, xpsd, ypsd, zpsd, drivepsd, freq_drive, aux]

def psd_in_folders(list_of_psd_folders, file_list_charge):
    arg = getdata(file_list_charge[0])[6]
    
    L = [[] for i in range(len(list_of_psd_folders))] # list of list

    for i in range(len(list_of_psd_folders)):
        xaux = 0
        yaux = 0
        zaux = 0
        file_list_psd = glob.glob(list_of_psd_folders[i]+"\*.h5")
        file_list_psd = list_file_time_order(file_list_psd)
        for j in file_list_psd:
            print j
            f, x, y, z, d, fd, arg_nouse = getdata(j)
            xaux = x + xaux
            yaux = y + yaux
            zaux = z + zaux
        xaux = np.sqrt(xaux/len(list_of_psd_folders[i]))
        yaux = np.sqrt(yaux/len(list_of_psd_folders[i]))
        zaux = np.sqrt(zaux/len(list_of_psd_folders[i]))
        L[i] = [f, xaux, yaux, zaux]
    return [L, arg]


def peaks(L, arg, tilt):
    if not len(L) == len(tilt):
        return "len tilt is wrong"
    X = []
    Y = []
    Z = []
    X2 = []
    Y2 = []
    Z2 = []
    for i in range(len(L)):
        xf = L[i][1][arg]
        yf = L[i][2][arg]
        zf = L[i][3][arg]
        x2f = L[i][1][2*arg-1]
        y2f = L[i][2][2*arg-1]
        z2f = L[i][3][2*arg-1]
        X.append(xf)
        Y.append(yf)
        Z.append(zf)
        X2.append(x2f)
        Y2.append(y2f)
        Z2.append(z2f)
    np.save(os.path.join(path_save,'2fxyz.txt'), (X2, Y2, Z2, tilt))
    fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(tilt, X, "b*", label="f")
    plt.plot(tilt, X2, "ro", label="2f")
    plt.ylabel("Vx/$\sqrt{Hz}$")
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(tilt, Y, "b*", label="f")
    plt.plot(tilt, Y2, "ro", label="2f")
    plt.ylabel("Vy/$\sqrt{Hz}$")
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(tilt, Z, "b*", label="f")
    plt.plot(tilt, Z2, "ro", label="2f")
    plt.ylabel("Vz/$\sqrt{Hz}$")
    plt.xlabel("tilt motor #2")
    plt.legend()
    plt.grid()
    plt.tight_layout(pad = 0)
    plt.savefig(os.path.join(path_save,'f_and_2f_peaks.pdf'))


if len(list_of_psd_folders) == len(tilt):
    print "continue"

L, arg = psd_in_folders(list_of_psd_folders, file_list_charge)

peaks(L, arg, tilt)

plt.figure()
plt.loglog(L[0][0], L[0][1])
plt.loglog(L[0][0], L[0][2])
plt.loglog(L[0][0], L[0][3])
plt.plot(L[0][0][arg], L[0][1][arg], "ro")
plt.plot(L[0][0][arg], L[0][2][arg], "ro")
plt.plot(L[0][0][arg], L[0][3][arg], "ro")

plt.plot(L[0][0][2*arg-1], L[0][1][2*arg -1], "ro")
plt.plot(L[0][0][2*arg-1], L[0][2][2*arg -1], "ro")
plt.plot(L[0][0][2*arg-1], L[0][3][2*arg -1], "ro")

plt.xlim(45, 100)
plt.show()
