import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import numpy as np
import bead_util as bu

path = r'C:\data\20170622\bead4_15um_QWP\reality_test2'

make_plot_vs_time = True
use_as_script = True

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2 ** 19


def getdata(fname):
    print "Opening file: ", fname
    f = h5py.File(fname, 'r')
    dset = f['beads/data/pos_data']
    dat = numpy.transpose(dset)
    Fs = dset.attrs['Fsamp']
    dat = dat * 10. / (2 ** 15 - 1)
    x = dat[:, bu.xi]
    d = dat[:, bu.drive]
    xpsd, freqs = matplotlib.mlab.psd(x - numpy.mean(x), Fs=Fs, NFFT=NFFT)
    drive, freqs = matplotlib.mlab.psd(d - numpy.mean(d), Fs=Fs, NFFT=NFFT)
    return freqs, xpsd, drive


def plot_data_together(path):
    N = NFFT / 2 + 1
    X, driveX = (np.zeros(N) for i in range(2))
    file_list = bu.time_ordered_file_list(path)
    n = len(file_list)
    for file in file_list:
        freqs, xpsd, drive = getdata(file)
        X += xpsd
        driveX += drive
    X = np.sqrt(X / n)
    driveX = np.sqrt(driveX / n)
    plt.figure()
    plt.loglog(freqs, X)
    plt.loglog(freqs, driveX)
    plt.grid()
    plt.show()


if use_as_script:
    plot_data_together(path)
