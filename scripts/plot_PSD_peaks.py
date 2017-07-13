from correlation import outputThetaPosition, getGainAndACamp, num_electrons_in_sphere
from VoltagevsAmplitude import conversion # gives N/V
import h5py, matplotlib, os, re, glob
import matplotlib.pyplot as plt
from bead_util import xi, drive, time_ordered_file_list
import numpy as np

# Inputs
NFFT = 2 ** 17
make_psd_plot = False
debugging = False
use_as_script = False

if use_as_script:
    calib = "/data/20170622/bead4_15um_QWP/charge9"
    path = "/data/20170622/bead4_15um_QWP/dipole27_Y"

if debugging:
    print "debugging on in plot_PSD_peaks.py: prepare for spam"
    print "num_electrons_in_sphere = ", num_electrons_in_sphere # 1E15
# in terminal, type 'python -m pdb plot_PSD_peaks.py'

def getdata(fname, give_squares = False):
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
    if give_squares: return freqs, xpsd, drivepsd # Hz, V^2/Hz, s
    else: return freqs, np.sqrt(xpsd), np.sqrt(drivepsd) # Hz, V/sqrtHz, 1/sqrtHz

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

def get_peak_amplitudes_Fernando(xpsd, dpsd):
    """ This is Fernando's weird way of averaging the peak amplitudes """
    a, d = get_positions(xpsd, dpsd)
    peaksD = dpsd[a] # amplitude of drive
    peaks2F = xpsd[d] + xpsd[d - 1] + xpsd[d + 1]  # all 2F peak bins
    return peaks2F/peaksD, d

def plot_peaks2Fernando(path, plot_peaks = True):
    file_list = time_ordered_file_list(path)
    amplitudes = []
    theta = []
    y_or_z = ""
    if make_psd_plot: plt.figure()
    for f in file_list:
        freqs, xpsd, dpsd = getdata(f)
        amp, i = get_peak_amplitudes_Fernando(xpsd, dpsd)
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

#"""               # this is Fernando's plot             """
#plot_peaks2Fernando(path)
#""""""""""""""""""""""" THINGS HERE """""""""""""""""""""""

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
        freqs, xpsd, drivepsd = getdata(f) # Hz, V/sqrtHz, 1/sqrtHz
        freq_pos, twice_freq_pos = get_positions(xpsd, drivepsd) # bins
        nx2.append(conversion*(xpsd[twice_freq_pos])/(drivepsd[freq_pos])) # N
    if use_theta:
        return np.array(theta), np.array(nx2)
    else:
        return np.array(nx2) # Newtons

def getConstant(calibration_path):
    """ normalization to units of electrons """
    calibration_list = time_ordered_file_list(calibration_path)
    i = min(len(calibration_list), 20)
    nx2 = get_PSD_peak_parameters(calibration_list[:i], use_theta = False) # for one electron
    return np.average(nx2)# Newtons/electron

def plot_PSD_peaks(path, calib_path, last_plot = False):
    file_list = time_ordered_file_list(path)
    c = getConstant(calib_path) # Newtons/electron
    if debugging:
        print "c = ", c
    theta, nx2 = get_PSD_peak_parameters(file_list) # steps, Newtons
    nx2 = nx2/c # electrons
    plt.figure()
    plt.plot(theta, nx2, 'o')
    plt.xlabel('Steps in theta')
    plt.ylabel('Amplitude [electrons]')
    plt.title('PSD peaks at twice the drive frequency')
    plt.grid()
    plt.show(block = last_plot)
    return

#plot_PSD_peaks(path, calib)

# now on to doing the area calibration thing
# integrating over basically the main peak
def get_area(f):
    w, x, d = getdata(f, give_squares=True) # Hz, V^2/Hz, 1/Hz
    binF = w[1] - w[0] # Hz
    gain, ACamp = getGainAndACamp(f) # unitless, V
    if debugging:
        fname = f[f.rfind('_')+1:f.rfind('.')]
        print ""
        print "DEBUGGING: get_area of ", fname
        print "           len(x) = ", len(x)
    i = np.argmax(d)
    if debugging:
        print "           i = ", i
        print ""
    x_in_Newtons = conversion*x*binF/(gain*ACamp)
    return sum(x_in_Newtons[i-2:i+3]) # Newtons

def peak_areas(path, c = 1, use_theta = False):
    a = [] # in units of Newtons
    x = []
    y_or_z = ""
    if debugging:
        print ""
        print "DEBUGGING: peak_areas"
        if use_theta: print "           using theta"
    file_list = glob.glob(path + "/*.h5")
    if debugging:
        i = min(len(file_list), 20)
        file_list = file_list[:i]
    if len(file_list) == 1:
        return np.array([0]), np.array([get_area(file_list[0])])/c
    for f in file_list:
        if debugging:
            print "           reading file ", f[len(path):], "inside peak_areas"
        a.append(get_area(f))
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
    one_electron = np.average(a[:i])
    return one_electron*num_electrons_in_sphere

def get_area_parameters(path, calib_path, use_theta = False):
    print "calibrating from ", calib_path
    c = calibration_area(calib_path)
    print "c = ", c
    print "finding areas for ", path
    x, a = peak_areas(path, c = c, use_theta = use_theta)
    return x, a # noise floor

def plot_areas(path, calib_path, use_theta = False, last_plot = False):
    x, a = get_area_parameters(path, calib_path, use_theta)
    print "Noise floor is at ", np.average(a), "fractions of an electron charge"
    print ""
    plt.figure()
    plt.loglog(x, a, 'o')
    if use_theta:
        plt.xlabel('Steps in theta')
    else:
        plt.xlabel('time [s]')
    plt.ylabel('peak area [electrons]')
    plt.title('Areas of PSD peaks at twice the drive frequency')
    plt.grid()
    plt.show(block = last_plot)
    return

# BLARG try again

def getdata_areas(fname, need_drive = True):
    f = h5py.File(fname, 'r')
    dset = f['beads/data/pos_data']
    dat = np.transpose(dset) * 10. / float(2 ** 15 - 1) # V
    Fs = dset.attrs['Fsamp']
    gain, ACamp = getGainAndACamp(fname) # unitless, V
    if debugging:
        print "           gain = ", gain, " and ACamp = ", ACamp
    x = (dat[:, 0] - np.mean(dat[:, 0]))/(gain*ACamp)
    xpsd, freqs = matplotlib.mlab.psd(x, Fs=Fs, NFFT=NFFT)
    if need_drive:
        drive_data = dat[:, 7] - np.mean(dat[:, 7])
        normalized_drive = drive_data / np.max(np.abs(drive_data))
        drivepsd, freqs = matplotlib.mlab.psd(normalized_drive, Fs=Fs, NFFT=NFFT)
        return freqs, xpsd, drivepsd # Hz, V^2/Hz, s
    else:
        return freqs, xpsd # Hz, V^2/Hz

def get_tot_psd(path, get_dpsd = True, last_plot = False):
    if debugging:
        print "\nDEBUGGING: get_tot_psd(path)"
    file_list = glob.glob(path + "/*.h5")
    n = len(file_list)
    xpsd = []
    if get_dpsd: dpsd = []
    for f in file_list:
        if debugging:
            print "           looking at file ", f[len(path):-3]
        if get_dpsd:
            w, x, d = getdata_areas(f)
        else:
            w, x = getdata_areas(f, need_drive = False)
        binF = w[2] - w[1]
        if xpsd == []:
            xpsd = x*binF/n
            if get_dpsd: dpsd = d*binF/n
        else:
            xpsd += x*binF/n
            if get_dpsd: dpsd += d*binF/n
    if debugging:
        plt.figure()
        plt.loglog(w, xpsd, label = 'xpsd')
        if get_dpsd: plt.loglog(w, dpsd, label = 'dpsd')
        plt.legend()
        plt.show(block = last_plot)
    if get_dpsd: return xpsd, dpsd
    else: return xpsd

def get_averaged_area(path, get_drive = False, dpsd = [], side = False, last_plot = False):
    if debugging:
        print "\nDEBUGGING: get_averaged_area(path)"
    if dpsd == []:
        xpsd, dpsd = get_tot_psd(path, last_plot = last_plot)
    else:
        xpsd = get_tot_psd(path, get_dpsd = False, last_plot = last_plot)
    i = np.argmax(dpsd)
    j = i + 10
    area = float(sum(xpsd[i-2:i+3]))
    if side:
        side_area = float(sum(xpsd[j-2:j+3]))
    if debugging:
        print "           i = ", i
        print "           area = ", area
        if side:
            print "           side_area = ", side_area, "\n"
    if side:
        return np.sqrt(area), np.sqrt(side_area)
    elif get_drive:
        return np.sqrt(area), dpsd
    else:
        return np.sqrt(area)

def find_floor(data_path, calib_path, data_has_drive = True, last_plot = False):
    print "\nfind_floor"
    
    print "calibrating with ", calib_path
    if data_has_drive:
        c = get_averaged_area(calib_path, last_plot = False)
    else:
        c, dpsd = get_averaged_area(calib_path, get_drive = True, last_plot = False)
    print "average calibration peak area is ", c
    
    print "measuring from ", data_path
    if data_has_drive:
        x, sx = get_averaged_area(data_path, side = True, last_plot = last_plot)
    else:
        x, sx = get_averaged_area(data_path, dpsd = dpsd, side = True, last_plot = last_plot)    
    print "average data peak area is ", x, "\n"
    
    floor = x/(c*num_electrons_in_sphere)
    side_floor = sx/(c*num_electrons_in_sphere)
    print "peak is at ", floor, "fractions of an electron charge"
    print "floor is at ", side_floor, "fractions of an electron charge\n"
    
    return floor, side_floor

if use_as_script:
    #plot_areas(path, calib, use_theta = True)
    """"""""""""""""""""" Inputs """""""""""""""""""""
    ### calibration files
    calib1 = "/data/20170622/bead4_15um_QWP/charge11"

    ### this is where the noise files are pulled out
    path1 = "/data/20170622/bead4_15um_QWP/reality_test2"
    ans1 = find_floor(path1, calib1)
    print ans1

    #calib = "/data/20170711/bead7_15um_QWP/calibration"
    #path = "/data/20170711/bead7_15um_QWP/reality_test3"
    #find_floor(path, calib, data_has_drive = False, last_plot = True)
