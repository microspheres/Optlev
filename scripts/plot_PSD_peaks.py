from correlation import outputThetaPosition
import h5py, matplotlib, os, re, glob
import matplotlib.pyplot as plt
from bead_util import xi, drive
import numpy as np

NFFT = 2 ** 16
make_psd_plot = True
debugging = False

if debugging:
    print "debugging on in plot_PSD_peaks.py: prepare for spam"
# in terminal, type 'python -m pdb plot_PSD_peaks.py'

calib = "/data/20170622/bead4_15um_QWP/charge9"
path = "/data/20170622/bead4_15um_QWP/dipole27_Y"

def getdata(fname):
    f = h5py.File(fname, 'r')
    dset = f['beads/data/pos_data']
    dat = np.transpose(dset)
    Fs = dset.attrs['Fsamp']
    dat = dat * 10. / (2 ** 15 - 1)
    x = dat[:, xi] - np.mean(dat[:, xi])
    xpsd, freqs = matplotlib.mlab.psd(x, Fs=Fs, NFFT=NFFT)
    drive_data = dat[:, drive] - np.mean(dat[:, drive])
    normalized_drive = drive_data / np.max(drive_data)
    drivepsd, freqs = matplotlib.mlab.psd(normalized_drive, Fs=Fs, NFFT=NFFT)
    return freqs, np.sqrt(xpsd), np.sqrt(drivepsd)

def time_ordered_file_list(path):
    file_list = glob.glob(path + "/*.h5")
    file_list.sort(key=os.path.getmtime)
    return file_list

def get_positions(xpsd, dpsd):
    """ returns position of drive frequency and twice the drive frequency """
    tolerance = 3 # bins
    a = np.argmax(dpsd) # drive frequency bin
    b = 2*a
    if debugging:
        print ""
        print "DEBUGGING: get_positions"
        print "           len(xpsd) = ", len(xpsd)
        print "           a = ", a
        print "           b = ", b
    c = np.argmax(xpsd[b-tolerance:b+tolerance])
    if debugging:
        print "           c = ", c
    d = (b - tolerance) + c # twice drive frequency bin
    if debugging:
        print "           d = ", d
        print ""
    return a, d

def get_peak_amplitudes(xpsd, dpsd):
    """ This is Fernando's weird way of averaging the peak amplitudes """
    a, d = get_positions(xpsd, dpsd)
    peaksD = dpsd[a] # amplitude of drive
    peaks2F = xpsd[d] + xpsd[d - 1] + xpsd[d + 1]  # all 2F peak bins
    return peaks2F/peaksD, d

def plot_peaks2F(path, plot_peaks = True):
    file_list = time_ordered_file_list(path)
    amplitudes = []
    theta = []
    y_or_z = ""
    if make_psd_plot: plt.figure()
    for f in file_list:
        freqs, xpsd, dpsd = getdata(f)
        amp, i = get_peak_amplitudes(xpsd, dpsd)
        amplitudes.append(amp)
        tpos, y_or_z = outputThetaPosition(f, y_or_z)
        theta.append(tpos)
        if make_psd_plot:
            plt.loglog(freqs, xpsd)
            plt.plot(freqs[i], xpsd[i], "x")
    if plot_peaks:
        plt.figure()
        plt.plot(theta, amplitudes, 'o')
        plt.grid()
        plt.show(block = False)
    return

# this is Fernando's plot
plot_peaks2F(path)

# this is Sumita's plot
def get_PSD_peak_parameters(file_list, use_theta = True):
    """ returns theta and ratio of [response at 2f] and [drive at f] """
    if use_theta:
        theta = []
        y_or_z = ""
    nx2 = []
    for f in file_list:
        if use_theta:
            tpos, y_or_z = outputThetaPosition(f, y_or_z)
            theta.append(tpos)
        freqs, xpsd, drivepsd = getdata(f) # psds in V/sqrtHz
        freq_pos, twice_freq_pos = get_positions(xpsd, drivepsd) # bins
        nx2.append((xpsd[twice_freq_pos])/(drivepsd[freq_pos])) # ratio of x/drive psds
    if use_theta:
        return np.array(theta), np.array(nx2)
    else:
        return np.array(nx2)

def getConstant(calibration_path):
    """ normalization to units of electrons """
    calibration_list = time_ordered_file_list(calibration_path)
    i = min(len(calibration_list), 20)
    nx2 = get_PSD_peak_parameters(calibration_list[:i], use_theta = False)
    return np.average(nx2)

def plot_PSD_peaks(path, calib_path, last_plot = False):
    file_list = time_ordered_file_list(path)
    c = getConstant(calib_path)
    if debugging:
        print "c = ", c
    theta, nx2 = get_PSD_peak_parameters(file_list)
    nx2 = nx2/c
    plt.figure()
    plt.plot(theta, nx2, 'o')
    plt.xlabel('Steps in theta')
    plt.ylabel('Amplitude [# electrons]')
    plt.title('PSD peaks at twice the drive frequency')
    plt.grid()
    plt.show(block = last_plot)
    return

plot_PSD_peaks(path, calib)

# now on to doing the area calibration thing
# integrating over basically the main peak
half_peak_width = 2 # bins

def get_area(f, i, j):
    w, x, d = getdata(f)
    n = len(x)
    if debugging:
        fname = f[f.rfind('_')+1:f.rfind('.')]
        print ""
        print "DEBUGGING: get_area of ", fname
        print "           len(x) = ", n
    if (i == 0 and j == 0) or (i > n/2): i, j = get_positions(x,d)
    if debugging:
        print "           i = ", i
        print "           j = ", j
        print ""
    return sum(x[j-half_peak_width:j+1+half_peak_width]), i, j

def peak_areas(path, c = 1, use_theta = False):
    a = []
    x = []
    y_or_z = ""
    i = 0
    j = 0
    if debugging:
        print ""
        print "DEBUGGING: peak_areas"
        if use_theta: print "           using theta"
    for f in glob.glob(path + "/*.h5"):
        if debugging:
            print "           reading file ", f, "inside peak_areas"
        area, i, j = get_area(f, i, j)
        a.append(area)
        if use_theta:
            tpos, y_or_z = outputThetaPosition(f, y_or_z)
            x.append(tpos)
        else:
            x.append(int(f[f.rfind('_')+1:f.rfind('.')]))
    x, a = zip(*sorted(zip(x, a))) # sort by time
    if debugging:
        print "           peak_areas worked!"
        print ""
    return np.array(x), np.array(a)/c

def calibration_area(calib_path):
    x, a = peak_areas(calib_path)
    i = min(len(a), 20) # take first few files
    if debugging:
        print ""
        print "DEBUGGING: calibration_area"
        print "           i = ", i
        print ""
    return np.average(a[:i])

def plot_areas(path, calib_path, use_theta = False):
    c = calibration_area(calib_path)
    if debugging:
        print ""
        print "DEBUGGING: plot_areas"
        print "           c = ", c
        print ""
    x, a = peak_areas(path, use_theta = use_theta, c = c)
    plt.figure()
    plt.plot(x, a, 'o')
    if use_theta:
        plt.xlabel('Steps in theta')
    else:
        plt.xlabel('time [s]')
    plt.ylabel('peak area [electrons]')
    plt.title('Areas of PSD peaks at twice the drive frequency')
    plt.grid()
    plt.show()
    return

plot_areas(path, calib, True)
