import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

path_list = [r"C:\data\201908020\22um_SiO2_pinhole\4\approach\2", r"C:\data\201908020\22um_SiO2_pinhole\4\approach\3", r"C:\data\201908020\22um_SiO2_pinhole\4\approach\4", r"C:\data\201908020\22um_SiO2_pinhole\4\approach\4", r"C:\data\201908020\22um_SiO2_pinhole\4\approach\5", r"C:\data\201908020\22um_SiO2_pinhole\4\approach\6", r"C:\data\201908020\22um_SiO2_pinhole\4\approach\7", r"C:\data\201908020\22um_SiO2_pinhole\4\approach\8", r"C:\data\201908020\22um_SiO2_pinhole\4\approach\9", r"C:\data\201908020\22um_SiO2_pinhole\4\approach\10",]

file_list = []
for i in path_list:
    f = glob.glob(i+"\*.h5")
    file_list.append(f)


NFFT = 2**13

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

	return [freqs, xpsd, ypsd, zpsd]


def plot(file_list, f1, f2):
    X = []
    Y = []
    Z = []
    F = []
    name = []
    plt.figure()
    for i in file_list:
        print i[0]
        f = 0*(getdata(i[0])[0])
        x = f
        y = f
        z = f
        for j in i:
            a = getdata(j)
            f = a[0]
            x = a[1] + x
            y = a[2] + y
            z = a[3] + z
        x = x/len(i)
        y = y/len(i)
        z = z/len(i)
        X.append(x)
        Y.append(y)
        Z.append(z)
        F.append(f)
        name.append(i)

        aa = np.where(f >= 1)[0][0]
        bb = np.where(f >= 150)[0][0]
        
        plt.subplot(3, 1, 1)
        plt.loglog(f[aa:bb], np.sqrt(x)[aa:bb],label=i)
        plt.ylabel("V/rtHz")
        # plt.legend(loc=3)
        plt.subplot(3, 1, 2)
        plt.loglog(f[aa:bb], np.sqrt(y)[aa:bb])
        plt.ylabel("V/rtHz")
        plt.legend(loc=3)
        plt.subplot(3, 1, 3)
        plt.loglog(f[aa:bb], np.sqrt(z)[aa:bb])
        plt.ylabel("V/rtHz")
        plt.legend(loc=3)
        plt.xlabel("Freq [Hz]")

    print name[0]
    plt.figure()
    for j in range(len(F)):
        c1 = np.where(F[j] >= f1)[0][0]
        c2 = np.where(F[j] >= f2)[0][0]
        area_X = np.sqrt(np.sum(X[j][c1:c2]))
        plt.semilogy(j, area_X, "ro")
        plt.ylabel("X Volts")
    plt.grid()
    plt.legend(loc=3)


    return [f, X, Y, Z, name]

plot(file_list, 33, 43)
plt.show()
