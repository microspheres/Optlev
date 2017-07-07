import h5py, matplotlib, os, re, glob
import matplotlib.pyplot as plt
import bead_util as bu
import numpy as np

NFFT = 2 ** 16
sleep = 5.
make_psd_plot = True

path = "/data/20170622/bead4_15um_QWP/dipole27_Y"

def getdata(fname):
    f = h5py.File(fname, 'r')
    dset = f['beads/data/pos_data']
    dat = np.transpose(dset)
    Fs = dset.attrs['Fsamp']
    dat = dat * 10. / (2 ** 15 - 1)
    x = dat[:, bu.x] - np.mean(dat[:, bu.x])
    xpsd, freqs = matplotlib.mlab.psd(x, Fs=Fs, NFFT=NFFT)
    drive = dat[:, bu.drive] - np.mean(dat[:, bu.drive])
    normalized_drive = drive / np.max(drive)
    drivepsd, freqs = matplotlib.mlab.psd(normalized_drive, Fs=Fs, NFFT=NFFT)
    return freqs, xpsd, drivepsd

def time_ordered_file_list(path):
    file_list = glob.glob(path + "\*.h5")
    file_list.sort(key=os.path.getmtime)
    return file_list

def get_positions(xpsd, dpsd):
    """ returns position of drive frequency and twice the drive frequency """
    tolerance = 3 # bins
    a = np.argmax(dpsd)
    b = 2*a
    c = np.argmax(xpsd[b-tolerance:b+tolerance])
    d = (b - tolerance) + c
    return a, d

def get_peak_amplitudes(xpsd, dpsd):
    a, d = get_positions(xpsd, dpsd)
    peaksD = dpsd[a]
    peaks2F = xpsd[d] + xpsd[d - 1] + xpsd[d + 1]  # all peak
    return peaksD, peaks2F, d

def plot_peaks2F(path, plot_peaks = True):
    file_list = time_ordered_file_list(path)
    peaks2F = np.zeros(len(file_list))
    peaksD = np.zeros(len(file_list))
    thetaY = np.zeros(len(file_list))
    thetaZ = np.zeros(len(file_list))
    if make_psd_plot: plt.figure()
    for i in range(len(file_list)):
        f = file_list[i]
        freqs, xpsd, dpsd = getdata(f)
        dp, fp, b = get_peak_amplitudes(xpsd, dpsd)
        peaksD[i] = dp
        peaks2F[i] = fp
        thetaY[i] = float(re.findall("-?\d+thetaY", f)[0][:-6])
        thetaZ[i] = float(re.findall("-?\d+thetaZ", f)[0][:-6])
        print thetaY[i], thetaZ[i]
        if make_psd_plot:
            plt.loglog(freqs, xpsd)
            plt.plot(freqs[b], xpsd[b], "x")
    peak2W = np.sqrt(peaks2F)
    peakD = np.sqrt(peaksD)
    if plot_peaks:
        plt.figure()
        plt.plot(thetaY, peak2W / peakD, 'o')
        plt.grid()
        plt.show(block = False)
    return thetaY, thetaZ, peak2W, peakD

# this is Fernando's plot
thetaY, thetaZ, peak2W, peakD = plot_peaks2F(path)

# this is Sumita's plot
def plot_PSD_peaks(path):
    file_list = time_ordered_file_list(path)
    freq = []
    freq2 = []
    x = []
    x2 = []
    for f in file_list:
        freqs, xpsd, drivepsd = getdata(f)
        freq_pos, twice_freq_pos = get_positions(xpsd, drivepsd)
        freq.append(freqs[freq_pos])
        x.append(freqs[freq_pos])
        freq2.append(freqs[twice_freq_pos])
        x2.append(freqs[twice_freq_pos])
    plt.figure()
    plt.plot(freq, x, 'o')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [V/sqrtHz]')
    plt.title('PSD peaks at drive frequency')
    plt.show(block = False)
    plt.figure()
    plt.plot(freq2, x2, 'o')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [V/sqrtHz]')
    plt.title('PSD peaks at twice the drive frequency')
    plt.show()

plot_PSD_peaks(path)
