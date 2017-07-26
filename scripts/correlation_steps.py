from scipy.optimize import curve_fit
import correlation, os, glob, h5py
from scipy.stats import mode, norm
import matplotlib.pyplot as plt
import bead_util as bu
import numpy as np

use_as_script = False


def getBackgroundDC(fname):
    i = fname.rfind('mVdc_') + 5
    j = fname.rfind('VDCbg')
    if 'mVDCbg' in fname:
        j = fname.rfind('mVDCbg')
        return float(fname[i:j]) / 1000.
    return float(fname[i:j])


def getData(fname, calib=False):
    """ assumes fname ends with a '.h5' """
    gain, ACamp = correlation.getGainAndACamp(fname)  # unitless, V
    fdrive = correlation.getFDrive(fname)  # Hz
    f = h5py.File(fname, 'r')
    dset = f['beads/data/pos_data']
    dat = np.transpose(dset)  # all this data is in volts
    x = dat[:, bu.xi]  # V
    Fs = dset.attrs['Fsamp']
    half_wavelength = int((Fs / fdrive) / 2.)  # bins
    x = x[:-half_wavelength]  # V
    x_data = ((x - np.average(x)) / float(len(x))) / (gain * ACamp)  # unitless
    if calib:
        drive0 = dat[:, bu.drive]  # V
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
    return zip(*sorted(zip(t, dc, corr)))


def formAveragedData(corr, dc):
    """ returns a dictionary with keys=DC offset, values=corr array """
    dcValues = map(abs, list(set(dc)))
    dc_corr_dict = {key: [] for key in dcValues}
    i = 0
    curr_dc_offset = 0.
    while i < len(corr):
        dcoff = dc[i]
        j = max(loc for loc, val in enumerate(dc) if val == dcoff) + 1
        curr_ave_corr = np.average(corr[i:j])
        if dcoff == -1.*curr_dc_offset: dc_corr_dict[abs(dcoff)]+=curr_ave_corr
        else: dc_corr_dict[dcoff].append(curr_ave_corr)
        curr_dc_offset = dcoff
        i = j
    return dc_corr_dict


def gaussian_distribution(x, A, u, sigma):
    return A * np.exp(-(x - u) ** 2 / (2 * sigma ** 2))


def plotGaussFit(data, make_plot=False):
    # get parameters
    n, bins = np.histogram(data, bins='auto')
    cfx = (bins[1:] + bins[:-1]) / 2.
    roughA = float(max(n))  # this is where I hard-code some rough estimates
    lbound = [roughA - 2., -5.e-18, 0.]
    ubound = [roughA + 2., 5.e-18, 5.e-18]
    popt, pcov = curve_fit(gaussian_distribution, cfx, n, bounds=(lbound, ubound))
    if not make_plot: return popt[1]
    perr = np.sqrt(np.diag(pcov))
    fitted_data = gaussian_distribution(cfx, *popt)
    mu, std = norm.fit(data)
    # print parameters
    print 'fitting to gaussian gives:'
    print '    mean = ', popt[1], ' with error ', perr[1]
    print '    standard deviation = ', popt[2], ' with error ', perr[2]
    print 'actual mean = ', mu
    print 'actual standard deviation = ', std
    # plot the figure
    plt.figure()
    plt.plot(cfx, fitted_data)
    plt.errorbar(cfx, n, yerr=np.sqrt(n), fmt='o')
    plt.show()


def plotGaussMean(dc_corr_dict):
    x, y = ([] for i in range(2))
    for v in dc_corr_dict.keys():
        x.append(v)
        y.append(plotGaussFit(dc_corr_dict[v]))
    plt.figure()
    plt.plot(x,y)
    plt.xlabel('DC offset voltages [V]')
    plt.ylabel('Mean correlation values [e]')
    plt.title('Mean correlation values vs DC offset')
    plt.show()


def fullPlotGaussMean(mpath, cpath):
    t, dc, corr = formData(mpath, cpath)
    dc_corr_dict = formAveragedData(corr, dc)
    plotGaussMean(dc_corr_dict)


if use_as_script:
    directory = "/data/20170717/bead15_15um_QWP/steps/"
    calibration_path = directory + "calibration_charge/"
    measurement_path = directory + "measurement_2/"

    t, dc, corr = formData(measurement_path, calibration_path)
    print "average correlation is ", float(sum(corr)) / float(len(corr))

    dc_corr_dict = formAveragedData(corr, dc)
    plotGaussMean(dc_corr_dict)


"""

average up step, average down step, add, go to next step
separate different dc values and find gaussian mean
plot gaussian mean wrt dc offset

"""