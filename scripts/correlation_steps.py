from scipy.optimize import curve_fit
import correlation, os, glob, h5py
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import mode
import bead_util as bu
import numpy as np

use_as_script = False
if use_as_script:
    directory = "/data/20170717/bead15_15um_QWP/steps/"
    calibration_path = directory + "calibration_charge/"
    measurement_path = directory + "measurement_2/"


def gauss(x, x0, y0, sigma):
    p = [x0, y0, sigma]
    return p[1] * np.exp(-((x - p[0]) / p[2]) ** 2)


def getBackgroundDC(fname):
    i = fname.rfind('mVdc_') + 5
    j = fname.rfind('VDCbg')
    if 'mVDCbg' in fname:
        j = fname.rfind('mVDCbg')
        return
    return float(fname[i:j])


def getData(fname, calib=False):
    """ assumes fname ends with a '.h5' """
    gain, ACamp = correlation.getGainAndACamp(fname) # unitless, V
    fdrive = correlation.getFDrive(fname) # Hz
    f = h5py.File(fname, 'r')
    dset = f['beads/data/pos_data']
    dat = np.transpose(dset)  # all this data is in volts
    x = dat[:, bu.xi] # V
    Fs = dset.attrs['Fsamp']
    half_wavelength = int((Fs / fdrive) / 2.) # bins
    x = x[:-half_wavelength] # V
    x_data = ((x - np.average(x)) / float(len(x))) / (gain * ACamp)  # unitless
    if calib:
        drive0 = dat[:, bu.drive] # V
        drive = drive0 - np.average(drive0)
        drive_data = drive / np.std(drive)  # normalized, unitless drive
        return x_data, drive_data
    else:
        time = dset.attrs['Time']
        bgDC = getBackgroundDC(fname)
        return x_data, bgDC, time


def calibrate(calibration_path, need_drive=True):
    """ goes through the x and drive data of each file (inputs)
        returns the index of the phase shift, the drive vector,
                and the normalization value of one electron """
    calibration_list = glob.glob(os.path.join(calibration_path + "*.h5"))
    N = float(len(calibration_list))
    phase_array, corr, drive = ([] for i in range(3))
    print "finding phase shift"
    for f in calibration_list:
        x_data, drive_data = getData(f, calib=True)
        # measure the correlation for normalization purposes
        corr_array = correlation.getCorrArray(x_data, drive_data)
        # here we average the drive and correlation arrays
        if drive == [] and corr == []:
            drive = drive_data / N
            corr = corr_array / N
        else:
            drive += drive_data / N
            corr += corr_array / N
        # index of largest correlation coefficient
        phase_array.append(np.argmax(corr_array))
    m, c = mode(phase_array)
    index = int(m[0])
    print "phase shift is ", index
    c = corr[index] * correlation.num_electrons_in_sphere  # V^2/electron
    print "calibrating constant c = ", c
    return index, c, np.array(drive)


def formData(mpath, cpath):
    index, c, drive_data = calibrate(cpath, need_drive=True)
    corr, dc, t = ([] for i in range(3))
    for f in glob.glob(os.path.join(mpath, "*.h5")):
        x_data, bgDC, time = getData(f)
        corr.append(correlation.correlate(x_data, drive_data, index, c))
        dc.append(bgDC)
        t.append(time)
    return zip*(sorted(zip(t, dc, corr)))


def formAveragedData(corr, dc):
    dcmag = map(abs, dc)
    dcValues = list(set(dcmag))
    corrValues = np.zeros(len(dcValues))
    for c, v in zip(corr, dcmag):
        i = dcValues.index(v)
        corrValues[i] += c
    corrValues = corrValues/float(len(corr))
    return zip(*sorted(zip(dcValues, corrValues)))


def plotAveragedData(corr, dc):
    d, c = formAveragedData(corr, dc)
    plt.figure()
    plt.plot(d, c, 'o')
    plt.xlabel('DC offset [V]')
    plt.ylabel('Averaged Correlation between drive and response [e]')
    plt.title('Averaged Correlation vs DC offset')
    plt.show(block=False)


def plotCorr(corr, dc, t):
    plt.figure()
    plt.plot(dc, corr, 'o')
    plt.xlabel('DC offset [V]')
    plt.ylabel('Correlation between drive and response [e]')
    plt.title('Correlation vs DC offset')
    plt.show(block=False)
    # now plot the correlations over time
    plt.figure()
    dc, t, corr = zip(*sorted(zip(dc, t, corr)))
    i = 0
    while i < len(corr):
        j = max(loc for loc, val in enumerate(dc) if val == dc[i]) + 1
        plt.plot(t[i:j], corr[i:j], 'o')
        i = j
    plt.xlabel('time [s]')
    plt.ylabel('Correlation between drive and response [e]')
    plt.title('Correlation vs. Time')
    plt.show()


# def plotGaussFit(corr):
#     corr = np.array(corr)
#     d = Counter(int(corr * 1e22))
#
#     >> > z = ['blue', 'red', 'blue', 'yellow', 'blue', 'red']
#     >> > Counter(z)
#     Counter({'blue': 3, 'red': 2, 'yellow': 1})
#
#     lists = sorted(d.items())  # sorted by key, return a list of tuples
#
#     x, y = zip(*lists)  # unpack a list of pairs into two tuples
#
#     plt.plot(x, y)
#     plt.show()
#
#     # Initialization parameters
#     p0 = [1., 1., 1.]
#     # Fit the data with the function
#     fit, tmp = curve_fit(gauss, x, y, p0=p0)
#
#     # Plot the results
#     plt.title('Fit parameters:\n x0=%.2e y0=%.2e sigma=%.2e' % (fit[0], fit[1], fit[2]))
#     # Data
#     plt.plot(x, y, 'r--')
#     # Fitted function
#     x_fine = np.linspace(xe[0], xe[-1], 100)
#     plt.plot(x_fine, gauss(x_fine, fit[0], fit[1], fit[2]), 'b-')
#     plt.savefig('Gaussian_fit.png')
#     plt.show()


if use_as_script:
    t, dc, corr = formData(measurement_path, calibration_path)
    print "average correlation is ", float(sum(corr))/float(len(corr))

    plotAveragedData(corr, dc)
    
    plotCorr(corr, dc, t)
