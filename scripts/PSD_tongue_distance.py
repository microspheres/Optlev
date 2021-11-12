import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob


folder_list = [r"C:\data\20190731\15um_SiO2\2\distances2\1", r"C:\data\20190731\15um_SiO2\2\distances2\2", r"C:\data\20190731\15um_SiO2\2\distances2\3", r"C:\data\20190731\15um_SiO2\2\distances2\4", r"C:\data\20190731\15um_SiO2\2\distances2\5", r"C:\data\20190731\15um_SiO2\2\distances2\6", ]

distances = np.array([11., 12., 13., 15., 17., 19.])/(2.23) # 2.23 is pixel/um

error_distances = np.sqrt((1/2.27)**2)

print distances

NFFT = 2**14

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

        print pid

	return [freqs, xpsd, ypsd, dat, zpsd]


def get_files(folder_list):
    F = []
    for i in folder_list:
        f = glob.glob(i+"\*.h5")
        F.append(f[0])
    return F

def get_psd(folder_list):
    F = get_files(folder_list)
    G = []
    for i in F:
        g = getdata(i)
        G.append(g)
    return [G, F]

def plot_all(folder_list):
    G, F = get_psd(folder_list)
    freq = G[0][0]
    plt.figure()
    for i in range(len(G)):
        plt.loglog(freq, np.sqrt(G[i][1]), label = str(F[i]))
        plt.ylabel("V/sqrt")
        plt.xlabel("freq [Hz]")
        plt.grid()
        plt.legend()
    return G

def area_all(folder_list, f1, f2, distances, error_distances):
    G = plot_all(folder_list)
    freq = G[0][0] 
    i1 = int(np.where( freq >= f1 )[0][0])
    i2 = int(np.where( freq >= f2 )[0][0])
    V = []
    for i in G:
        v = np.sqrt(np.sum(i[1][i1:i2]))
        V.append(v)

    plt.figure()
    plt.grid()
    plt.ylabel("Volts [V]")
    plt.xlabel("Distance [$\mu m$]")
    a = 0
    for i in V:
        plt.errorbar(distances[a], i, xerr = error_distances, fmt = "ro")
        a = a + 1

    return V
        

        
area_all(folder_list, 10., 35., distances, error_distances)
plt.show()

# folder_list = [r"C:\data\20190731\15um_SiO2\2\distances\1", r"C:\data\20190731\15um_SiO2\2\distances\2", r"C:\data\20190731\15um_SiO2\2\distances\3", r"C:\data\20190731\15um_SiO2\2\distances\4", r"C:\data\20190731\15um_SiO2\2\distances\5", r"C:\data\20190731\15um_SiO2\2\distances\6", r"C:\data\20190731\15um_SiO2\2\distances\7"]

# distances = np.array([9., 11., 13., 17., 19., 21., 7. ])/(2.27) # 2.27 is pixel/um

# error_distances = np.sqrt((1/2.27)**2)
