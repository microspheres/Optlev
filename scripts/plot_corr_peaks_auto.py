from plot_PSD_peaks import plot_peaks2Fernando
from charge import get_most_recent_file
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import bead_util as bu
import os, re, h5py
import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import numpy as np
import os, time

fdrive = 47.
Fs = 10000
li = 45.
ls = 49.
butterp = 3
boundi = 1500
bounds = 7500
make_psd_plot = True
path = r"C:\data\20170717\bead15_15um_QWP\dipole18_Y"


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def getdata_x_d(fname):
    _, fext = os.path.splitext(fname)
    if (fext == ".h5"):
        f = h5py.File(fname, 'r')
        dset = f['beads/data/pos_data']
        dat = np.transpose(dset)
        Fs = dset.attrs['Fsamp']
        dat = dat * 10. / (2 ** 15 - 1)
    else:
        dat = np.loadtxt(fname, skiprows=5, usecols=[2, 3, 4, 5, 6])

    x = dat[:, 0] - np.mean(dat[:, 0])
    x = butter_bandpass_filter(x, li, ls, Fs, butterp)
    driveN = ((dat[:, bu.drive] - np.mean(dat[:, bu.drive]))) / np.max((dat[:, bu.drive] - np.mean(dat[:, bu.drive])))

    driveNf = butter_bandpass_filter(driveN, li, ls, Fs, butterp) / np.max(
        butter_bandpass_filter(driveN, li, ls, Fs, butterp))

    drive2W = (driveNf * driveNf - np.mean(driveNf * driveNf)) / np.max(driveNf * driveNf - np.mean(driveNf * driveNf))

    return [x, driveNf, drive2W]


def corr_aux(drive2WN, driveN, x):
    zero = np.zeros(0)
    shift_x = np.append(x, zero)
    shift_d = np.append(zero, driveN)
    shift_d2W = np.append(zero, drive2WN)

    fftx = np.fft.rfft(shift_x)
    fftd = np.fft.rfft(shift_d)
    fftd2W = np.fft.rfft(shift_d2W)

    Fi = np.argmax(fftd2W) - 3
    Fs = np.argmax(fftd2W) + 3
    jx = 1

    corr = np.sum(np.conjugate(fftd[boundi:bounds]) * fftx[boundi:bounds] / jx[boundi:bounds]) / np.sum(
        np.conjugate(fftd[boundi:bounds]) * fftd[boundi:bounds] / jx[boundi:bounds])
    corr = corr

    corr2W = np.sum(np.conjugate(fftd2W[Fi:Fs]) * fftx[Fi:Fs])
    corr2W = corr2W

    return [corr, corr2W]


def plot_peaks2F(path):
    file_list = bu.time_ordered_file_list(path)
    corr2F = np.zeros(len(file_list))
    corrF = np.zeros(len(file_list))
    thetaY = np.zeros(len(file_list))
    thetaZ = np.zeros(len(file_list))
    for i in range(len(file_list)):
        x, d, d2 = getdata_x_d(file_list[i])
        corra, corr2a = corr_aux(d2, d, x, 0)
        corrF[i] = corra
        corr2F[i] = corr2a
        f = file_list[i]
        thetaY[i] = float(re.findall("-?\d+thetaY", f)[0][:-6])
        thetaZ[i] = float(re.findall("-?\d+thetaZ", f)[0][:-6])
        print thetaY[i], thetaZ[i]
        corrF[i] = np.correlate(x, d)
    return [thetaY, thetaZ, corrF, corr2F]


thetaY, thetaZ, corrW, corr2W = plot_peaks2F(path)

plt.figure()
plt.plot(thetaY, corrW, 'o')
plt.grid()
plt.show()

last_file = ""
while (True):
    ## get the most recent file in the directory and calculate the correlation

    cfile = get_most_recent_file(path)

    ## wait a sufficient amount of time to ensure the file is closed
    print cfile
    time.sleep(ts)

    if (cfile == last_file):
        continue
    else:
        last_file = cfile

    ## this ensures that the file is closed before we try to read it
    time.sleep(1)

    if (not best_phase):
        best_phase = getphase(cfile)

    corr = getdata(cfile, best_phase)
    corr_data.append(corr)

    if make_plot:
        plt.plot(np.array(corr_data))
        plt.draw()
        plt.pause(0.001)
        plt.grid()
