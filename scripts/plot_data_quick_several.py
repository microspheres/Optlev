import numpy, h5py
from matplotlib.mlab import psd
import matplotlib.pyplot as plt
import numpy as np
import bead_util as bu

path = r'C:\data\20170717\bead15_15um_QWP\reality_test_Wed_night_step'
NFFT = 2 ** 19


def getdata(fname):
    print "Opening file: ", fname
    f = h5py.File(fname, 'r')
    dset = f['beads/data/pos_data']
    dat = numpy.transpose(dset) * 10. / (2. ** 15 - 1.)
    Fs = dset.attrs['Fsamp']
    xpsd, freqs = psd(dat[:, bu.xi] - numpy.mean(dat[:, bu.xi]), Fs=Fs, NFFT=NFFT)
    return freqs, xpsd


def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


def get_data_together(file_list):
    N = NFFT / 2 + 1
    X = 0
    n = len(file_list)
    for fname in file_list:
        i = fname.find('synth')
        j = fname.find('mV', i) + 2
        k = fname.rfind('Hz')
        drive_freq = int(fname[j:k])
        freqs, xpsd = getdata(fname)
        index = find_nearest(freqs, drive_freq)
        X += xpsd[index]
    return np.sqrt(X / n)


def get_averaged_data(path):
    file_list = bu.time_ordered_file_list(path)
    n = len(file_list)
    X = np.zeros(n)
    for i in range(n):
        X[i] = get_data_together(file_list[i:i+1])
    return X


def plot_averaged_data(path):
    plt.figure()
    plt.plot(get_averaged_data(path))
    plt.show()
    return


plot_averaged_data(path)
