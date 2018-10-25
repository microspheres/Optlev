from os.path import join, splitext
from h5py import File
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.mlab import psd as get_psd
import numpy as np
from bead_util import xi, yi

folder_list = [r"C:\data\20180817\transfer_function_before_chamber_X",
               r"C:\data\20180817\transfer_function_before_chamber_Y",
               r"C:\data\20180817\transfer_function_before_chamber_0",
               r"C:\data\20180816\transfer_function_0_gain_after_chamber",
               r"C:\data\20180816\transfer_function_y_gain_0.01_after_chamber",
               r"C:\data\20180816\transfer_function_x_gain_0.01_after_chamber"]

xb_index = 0
xa_index = 5

yb_index = 1
ya_index = 4

Fs = 10e3  # this is ignored with HDF5 files
NFFT = 2 ** 19


def getdata(fname):
    print "Opening file: ", fname
    ## guess at file type from extension
    _, fext = splitext(fname)
    if (fext == ".h5"):
        f = File(fname, 'r')
        dset = f['beads/data/pos_data']
        dat = np.transpose(dset)
        max_volt = 10.
        nbit = 2 ** 15 - 1
        Fs = dset.attrs['Fsamp']
        dat = 1.0 * dat * max_volt / nbit
    else:
        dat = np.loadtxt(fname, skiprows=5, usecols=[2, 3, 4, 5, 6])

    xpsd, freqs = get_psd(dat[:, xi] - np.mean(dat[:, xi]), Fs=Fs, NFFT=NFFT)
    ypsd, freqs = get_psd(dat[:, yi] - np.mean(dat[:, yi]), Fs=Fs, NFFT=NFFT)
    return freqs, xpsd, ypsd


def make_measurement_npy_files(folder):
    file_list = glob(folder + "\*.h5")
    n = float(len(file_list))
    if n == 0.:
        return "error empty folder"

    freq, xp2, yp2 = getdata(file_list[0])
    for i in file_list[1:]:
        f, xaux, yaux = getdata(i)
        xp2 = xp2 + xaux
        yp2 = yp2 + yaux
    xp2 = xp2 / n
    yp2 = yp2 / n

    np.save(join(folder, "measurement_x"), [freq, xp2])
    np.save(join(folder, "measurement_y"), [freq, yp2])
    return freq, xp2, yp2


def load_psds(folder):
    xfreq, xpsd = np.load(join(folder, "measurement_x.npy"))
    yfreq, ypsd = np.load(join(folder, "measurement_y.npy"))
    return xfreq, xpsd, yfreq, ypsd


def read_psd_folders(list_of_folders):
    xpsd2s = []
    xfreqs = []
    ypsd2s = []
    yfreqs = []
    folder_name = []
    for f in list_of_folders:
        xfreq, xpsd2, yfreq, ypsd2 = load_psds(f)
        xpsd2s.append(xpsd2)
        xfreqs.append(xfreq)
        ypsd2s.append(ypsd2)
        yfreqs.append(yfreq)
        folder_name.append(str(f))
    return xfreqs, xpsd2s, yfreqs, ypsd2s, folder_name


def get_peak_position(freqs, psd):
    """
    assumes measurement frequency is between 10 and 100 Hz
    :param freqs: 
    :param psd: 
    :return peak_position: 
    """
    position_of_1 = np.argmin(np.abs(freqs - np.ones(len(freqs)) * 10.))
    position_of_2 = np.argmin(np.abs(freqs - np.ones(len(freqs)) * 100.))
    peak_position = np.argmax(psd[position_of_1:position_of_2]) + position_of_1
    return peak_position


def plot_one(f, p2, arg, name, channel_name):
    plt.figure()
    for i in range(len(folder_list)):
        curr_label = name[i]
        curr_label = curr_label[curr_label.rfind('\\', 16) + 1:]
        plt.loglog(f[i], np.sqrt(p2[i]), label=curr_label)
    loc_of_line = f[0][arg]
    plt.axvline(loc_of_line, color="r", linestyle="--")
    plt.legend()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V/$\sqrt{Hz}$]" + " of channel " + channel_name)
    plt.xlim(loc_of_line - 1, loc_of_line + 1)
    plt.grid()
    plt.tight_layout(pad=0)


def plot_all(list_of_folders, actually='yes'):
    xf, xp2, yf, yp2, name = read_psd_folders(list_of_folders)
    arg = get_peak_position(xf[0], xp2[0])
    if actually[0] == 'y':
        plot_one(xf, xp2, arg, name, 'X')
        plt.show(block=False)
        plot_one(yf, yp2, arg, name, 'Y')
        plt.show()
    return arg, xp2, yp2


# make_measurement_npy_files(folder_list[yb_index])

peak_index, xpsds, ypsds = plot_all(folder_list, 'yeah')

xb = xpsds[xb_index]
xa = xpsds[xa_index]
yb = ypsds[yb_index]
ya = ypsds[ya_index]

print "the transfer function for x is " + str(xb[peak_index] / xa[peak_index])
print "                         as in " + str(xa[peak_index] / xb[peak_index])

print "the transfer function for y is " + str(yb[peak_index] / ya[peak_index])
print "                         as in " + str(ya[peak_index] / yb[peak_index])
