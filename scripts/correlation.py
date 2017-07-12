import numpy as np
import matplotlib.pyplot as plt
import os, glob, h5py
import bead_util as bu

""""""""""""""""""""" Inputs """""""""""""""""""""
### calibration files
calib1 = "/data/20170622/bead4_15um_QWP/charge9"
calib2 = "/data/20170622/bead4_15um_QWP/arb_charge"

### this is where the noise files are pulled out
path1 = "/data/20170622/bead4_15um_QWP/reality_test2"
path2 = "/data/20170622/bead4_15um_QWP/reality_test3"

fdrive = 41. # Hz
#wavelength = Fs/fdrive because Fs = samples/second

make_calibration_plot = False # do we want to see the step plot?
plot_fft = False # do we want to see the fft of the correlation plots?

debugging = False # are we in debugging mode?
# in terminal, type 'python -m pdb correlation.py'

# Calculate # electrons
sphere_diameter = 15 # micron
sphere_radius = sphere_diameter/2000000. # meters
sphere_volume = (4./3.)*np.pi*sphere_radius**3 # m^3
density = 2196. # kg/m^3
sphere_mass = sphere_volume*density # kg
molecular_mass = 9.9772E-26 # kg/molecule
num_electrons = 30 # electrons/molecule
num_electrons_in_sphere = num_electrons*sphere_mass/molecular_mass # electrons
#                       = ~1E-15

""""""""""""""""""""" Code """""""""""""""""""""""
### List of files
calib_list1 = glob.glob(os.path.join(calib1, "*.h5"))
#calib_list2 = glob.glob(os.path.join(calib2, "*.h5"))
file_list1 = glob.glob(os.path.join(path1,"*.h5"))
#file_list2 = glob.glob(os.path.join(path2,"*.h5"))
if debugging:
    print "debugging turned on: prepare for spam"
    print ""

def sortFileList(file_list):
    N = len(file_list)
    if debugging:
        print ""
        print "DEBUGGING: sortFileList N = ", N
    if N == 1:
        if debugging:
            print ""
        return file_list
    else:
        new_list = ['']*N
        for i in range(N):
            f = file_list[i]
            j = f.rfind('_') + 1
            k = f.rfind('.')
            if debugging:
                print "           sortFileList n = ", f[j:k]
            n = int(f[j:k])
            new_list[n] = f
        if debugging:
            print ""
        return new_list

def outputThetaPosition(f, y_or_z):
    if y_or_z == "":
        i = f.find('stage_tilt_') + 11
        j = f.rfind('thetaY_')
        k = j + 7
        l = f.rfind('thetaZ')
        y = int(f[i:j])
        z = int(f[k:l])
        if debugging:
            print "           thetaY = ", y
            print "           thetaZ = ", z
        if y == 0 and z == 0:
            return 0, ""
        elif y == 0 and z != 0:
            return z, "z"
        else:
            return y, "y"
    elif y_or_z == "y":
        i = f.find('stage_tilt_') + 11
        j = f.rfind('thetaY_')
        y = int(f[i:j])
        if debugging:
            print "           thetaY = ", y
        return y, "y"
    elif y_or_z == "z":
        k = f.rfind('thetaY_') + 7
        l = f.rfind('thetaZ')
        z = int(f[k:l])
        if debugging:
            print "           thetaZ = ", z
        return z, "z"

def getGainAndACamp(fname):
    i = fname.rfind('cool_G') + 6
    j = fname.find('_', i)
    k = fname.rfind('synth') + 5
    l = fname.find('mV', k)
    gain = float(fname[i:j])
    ACamp = float(fname[k:l])/1000. # Volts
    return gain, ACamp

def getData(fname, get_drive = False, truncate_x = True):
    """ assumes fname ends with a '.h5' 
        returns unitless data from file """
    if debugging:
        print ""
        debug_file = fname[fname.rfind('synth'):fname.rfind('.')]
        print "DEBUGGING: getting data from ", debug_file
    gain, ACamp = getGainAndACamp(fname)
    if debugging:
        print "           gain of file ", debug_file, " = ", gain
        print "           AC amplitude of file ", debug_file, " = ", ACamp
    f = h5py.File(fname, 'r')
    dset = f['beads/data/pos_data']
    dat = np.transpose(dset) # all this data is in volts
    dat = dat/(gain*ACamp) # makes this data unitless
    x = dat[:,bu.xi]
    if truncate_x:
        Fs = dset.attrs['Fsamp']
        half_wavelength = int((Fs/fdrive)/2.)
        if debugging:
            print '           half_wavelength of ', debug_file, ' is ', half_wavelength
        x = x[:-half_wavelength]
    x_data = (x - np.average(x))/float(len(x)) # normalize for calibration
    if debugging:
        print "           got the x_data of ", debug_file
    if get_drive:
        drive0 = dat[:,bu.drive]
        drive = drive0 - np.average(drive0)
        drive_data = drive/np.std(drive) # normalized, unitless drive
        twice_drive0 = drive0*drive0
        twice_drive = twice_drive0 - np.average(twice_drive0)
        twice_drive_data = twice_drive/np.std(twice_drive)
        if debugging:
            print "           got the drive_data of ", debug_file
            print ""
        return x_data, drive_data, twice_drive_data
    else:
        time = dset.attrs['Time']
        if debugging:
            print ""
        return x_data, time

def getThetaData(fname, y_or_z):
    """ assumes fname ends with a '.h5' 
        returns unitless data from file """
    if debugging:
        print ""
        debug_file = fname[fname.find('synth'):fname.rfind('.')]
        print "DEBUGGING: getThetaData on ", debug_file
    # get data from the file name
    position, new_y_or_z = outputThetaPosition(fname, y_or_z)
    if debugging:
        print "           theta position of file ", debug_file, " = ", position
    # read the file
    x_data, drive_data, twice_drive_data = getData(fname, get_drive = True, truncate_x = False)
    if debugging:
        print ""
    return x_data, drive_data, twice_drive_data, position, new_y_or_z

def getCorrArray(x_data, drive_data, c = 1.):
    """ returns the normalized correlation between the drive and the response """
    corr_array = np.correlate(x_data, drive_data)
    if debugging:
        print ""
        print "DEBUGGING: getCorrArray() worked!"
        print ""
    return corr_array/c # response in terms of #e at the calibration field

def calibrate(calibration_list, make_plot = make_calibration_plot, last_plot = False):
    """ goes through the x and drive data of each file (inputs)
        returns the index of the phase shift (independant of normalization)
                and the normalization value of one electron
                (assuming calibration_data is of only one electron)
        assume calibration_data_list is sorted by measurement time"""
    if debugging:
        index = min(len(calibration_list), 20)
        calibration_list = calibration_list[:index]
        print ""
        print "DEBUGGING calibrate()"
    N = len(calibration_list)
    drive = []
    twice_drive = []
    phase_array = []
    dat_arrays = []
    t_dat_arrays = []
    print "finding phase shift"
    for i in range(N):
        if debugging:
            print "           ", i, "th iteration of for loop"
        x_data, drive_data, twice_drive_data = getData(calibration_list[i], get_drive= True)
        # here we average the drive "measurements"
        if drive == []:
            drive = drive_data/float(N)
        else:
            drive += drive_data/float(N)
        if twice_drive == []:
            twice_drive = twice_drive_data/float(N)
        else:
            twice_drive += twice_drive_data/float(N)
        # now measure the correlation for normalization purposes
        corr_array = getCorrArray(x_data, drive_data)
        dat_arrays.append(corr_array) # array of all the correlation arrays
        t_corr_array = getCorrArray(x_data, twice_drive_data)
        t_dat_arrays.append(t_corr_array) # array of the correlation arrays at 2f
        # only work with the first few files to get the phase and c
        if i < 20:
            phase_array.append(np.argmax(corr_array)) # index of largest correlation coefficient
    index = int(np.average(phase_array))
    print "phase shift is ", index
    corr = []
    for dat_arr in dat_arrays:
        corr.append(dat_arr[index])
    tcorr = []
    for t_arr in t_dat_arrays:
        tcorr.append(t_arr[index])
    i = min(len(corr), 20)
    c = np.average(corr[:i]) # V^2/electron
    print "calibrating constant c = ", c
    if debugging:
        print ""
    if make_plot:
        plot_calibration_data(corr, tcorr, c, last_plot)
    return index, c, drive, twice_drive

def plot_calibration_data(corr, tcorr, c, last_plot):
    """ plots the calibration data """
    print "plotting calibration data"
    plt.figure()
    plt.plot(np.array(corr)/c, 'o')
    plt.ylabel('Correlation [#e]')
    plt.title('Calibration data correlation')
    plt.show(block = False)
    print "plotting calibration data at twice the frequency"
    plt.figure()
    plt.plot(np.array(tcorr)/c, 'o')
    plt.ylabel('Correlation [#e]')
    plt.title('Calibration correlation at 2f')
    plt.show(block = last_plot)

def correlate(x_data, drive_data, index, coeff):
    arr = getCorrArray(x_data, drive_data, c = coeff)
    if debugging:
        print ""
        print "DEBUGGING: correlate() worked!"
        print ""
    return arr[index]

def getResponseArray(file_list):
    """ assumes file_list is in order """
    x_arr = []
    t_arr = []
    for fname in file_list:
        x, time = getData(fname)
        x_arr.append(x)
        t_arr.append(time)
    t_arr = np.array(t_arr)
    t_arr = t_arr - t_arr[0]
    if debugging:
        print ""
        print "DEBUGGING: the first element in t_arr is ", t_arr[0]
        print "           getResponceArray() worked!"
        print ""
    return x_arr, t_arr

def plotCorr(file_list, drive, twice_drive, index = 0, c = 1, last_plot = False, fft = plot_fft):
    """ takes in a measurement's data array and the phase shift
        returns a plot of the correlation """
    sorted_file_list = sortFileList(file_list)
    print "fetching noise data"
    if debugging:
        index = min(len(sorted_file_list), 20)
        sorted_file_list = sorted_file_list[:index]
    x_arr, t_arr = getResponseArray(sorted_file_list)
    of = []
    tf = []
    if debugging:
        print ""
        print "DEBUGGING: got response array in plotCorr()"
    for x in x_arr:
        if debugging:
            print "           iteration in plotCorr number ", len(of)
        of.append(correlate(x, drive, index, c))
        tf.append(correlate(x, twice_drive, index, c))
    print "plotting noise data"
    plt.figure()
    plt.plot(t_arr, of, 'o')
    plt.ylabel('Correlation [e]')
    plt.title('Correlation of drive and response')
    plt.show(block=False)
    print "plotting noise at twice the frequency"
    plt.figure()
    plt.plot(t_arr, tf, 'o')
    plt.ylabel('Correlation [e]')
    plt.title('Correlation of drive squared and response')
    if fft:
        plt.show(block = False) # for previous plot
        plt.figure()
        fourier = np.fft.rfft(of - np.average(of))
        freq = np.fft.rfftfreq(len(of), d = (t_arr[1] - t_arr[0]))
        thing_to_plot = np.square(np.absolute(fourier))
        if debugging:
            print "           length of correlation vector is ", len(of)
            print "           length of fft of correlation is ", len(fourier)
            print "           length of timestream is ", len(t_arr)
            print "           minimum value of fft is ", min(fourier)
            print "           minimum value of thing_to_plot is ", min(thing_to_plot)
            print "           d = ", (t_arr[1] - t_arr[0])
        #plt.loglog(freq, thing_to_plot)
        plt.plot(freq, thing_to_plot, 'o')
        plt.xscale('log')
        plt.title('fft of correlation')
    if debugging:
        print ""
    print "average correlation is ", np.average(of)
    plt.show(block = last_plot)

def full_correlation_plots(calib_list, file_list, last_plot = False):
    """ does everything: calibrates and makes correlation plots at 
        the drive frequency and at twice the drive frequency 
        in total produces 2-4 plots depending on calibration and fft """
    if debugging:
        print ""
        print "DEBUGGING: full_correlation_plots"
    # sort calib_list
    calibration_list = sortFileList(calib_list)
    # calibrate and maybe plot the step plot
    i, c, drive, twice_drive = calibrate(calibration_list)
    if debugging:
        print "           calibrated in full_correlation_plots"
    # plot the correlation at the drive frequency
    plotCorr(file_list, drive, twice_drive, i, c, last_plot = last_plot)
    if debugging:
        print "           plotted f and 2f in full_correlation_plots"
        print ""

def theta_correlation_plots(path, last_plot = False):
    """ does everything: calibrates and makes correlation plots at 
        the drive frequency and at twice the drive frequency 
        in total produces 2-4 plots depending on calibration and fft
        goes through the x and drive data of each file (inputs)"""
    if debugging:
        print ""
        print "DEBUGGING: theta_full_correlation_plots on ", path
    file_list = glob.glob(os.path.join(path, "*.h5"))
    if debugging:
        index = min(len(file_list), 20)
        file_list = file_list[:index]
    pos = []
    corr = []
    tcorr = []
    print "finding phase shift"
    y_or_z = ""
    for f in file_list:
        x_data, drive_data, twice_drive_data, position, y_or_z = getThetaData(f, y_or_z)
        pos.append(position)
        # now measure the correlation
        corr_array = getCorrArray(x_data, drive_data)
        corr.append(corr_array[0]) # array of all the correlation arrays
        t_corr_array = getCorrArray(x_data, twice_drive_data)
        tcorr.append(t_corr_array[0]) # array of the correlation arrays at 2f
    # now plot
    print "plotting noise data"
    plt.figure()
    plt.plot(pos, corr, 'o')
    plt.xlabel('Steps in the theta direction')
    plt.ylabel('Correlation [e]')
    plt.title('Correlation of drive and response')
    plt.show(block=False)
    print "plotting noise at twice the frequency"
    plt.figure()
    plt.plot(pos, tcorr, 'o')
    plt.xlabel('Steps in the theta direction')
    plt.ylabel('Correlation [e]')
    plt.title('Correlation of drive squared and response')
    plt.show(block=last_plot)

""" Now we plot """

full_correlation_plots(calib_list1, file_list1, last_plot = True)

#w_path = "/data/20170622/bead4_15um_QWP/dipole27_Y" # this has the W
#theta_correlation_plots(w_path, last_plot = True)

"""
FIRST
Take the calibration data files.
1. figure out the wavelength of the data in terms of sample points / bins
2. remove the first half-wavelength of data from the x array
3. np.correlate(drive, truncated_x_array)
4. take the argmax of that - make a vector and find the average of that index
5. plot this stuff to make sure you can see the steps
NEXT
Go to the noise files.
1. use the index as before as the phase shift, and do np.correlate again
2. get a number for all the files at that index
3. plot.
"""
