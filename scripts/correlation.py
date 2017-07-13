import numpy as np
import matplotlib.pyplot as plt
import os, glob, h5py
import bead_util as bu

""""""""""""""""""""" Inputs """""""""""""""""""""
use_as_script = False # run this file
plotting = False # do we want to see plots at all?
make_calibration_plot = False # do we want to see the step plot?
plot_fft = False # do we want to see the fft of the correlation plots?
# in terminal, type 'python -m pdb correlation.py'
debugging = False # are we in debugging mode?

""""""""""""""""""""" Code """""""""""""""""""""""
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
fdrive = 41. # Hz; wavelength = Fs/fdrive because Fs = samples/second
if debugging: print "debugging turned on: prepare for spam \n"

def basic_plot(x, of_corr, tf_tcorr, use_theta = False, modifier = '', last_plot = False):
    print "plotting " + modifier + " noise data"
    plt.figure()
    plt.plot(x, of_corr, 'o')
    if use_theta: plt.xlabel('Steps in the theta direction')
    plt.ylabel('Correlation [e]')
    plt.title(modifier + ' Correlation of drive and response')
    plt.show(block=False)
    print "plotting " + modifier + " noise at twice the frequency"
    plt.figure()
    plt.plot(x, tf_tcorr, 'o')
    if use_theta: plt.xlabel('Steps in the theta direction')
    plt.ylabel('Correlation [e]')
    plt.title(modifier + 'Correlation of drive squared and response')
    plt.show(block=last_plot)

def fft_plot(time_array, peak_amplitudes_at_f, last_plot = False):
    plt.figure()
    fourier = np.fft.rfft(peak_amplitudes_at_f - np.average(peak_amplitudes_at_f))
    freq = np.fft.rfftfreq(len(peak_amplitudes_at_f), d = (time_array[1] - time_array[0]))
    thing_to_plot = np.square(np.absolute(fourier))
    if debugging:
        print "           length of correlation vector is ", len(peak_amplitudes_at_f)
        print "           length of fft of correlation is ", len(fourier)
        print "           length of timestream is ", len(time_array)
        print "           minimum value of fft is ", min(fourier)
        print "           minimum value of thing_to_plot is ", min(thing_to_plot)
        print "           d = ", (time_array[1] - time_array[0])
    plt.plot(freq, thing_to_plot, 'o')
    plt.xscale('log')
    plt.title('fft of correlation')
    plt.show(block = last_plot)

def sortFileList(file_list):
    N = len(file_list)
    if debugging: print "\nDEBUGGING: sortFileList N = ", N
    if N == 1:
        if debugging: print ""
        return file_list
    else:
        new_list = ['']*N
        for i in range(N):
            f = file_list[i]
            j = f.rfind('_') + 1
            k = f.rfind('.')
            if debugging: print "           sortFileList n = ", f[j:k]
            n = int(f[j:k])
            new_list[n] = f
        if debugging: print ""
        return new_list

def outputThetaPosition(f, y_or_z):
    if y_or_z == "":
        i = f.find('stage_tilt_') + 11
        j = f.rfind('thetaY_')
        k = j + 7
        l = f.rfind('thetaZ')
        y = int(f[i:j])
        z = int(f[k:l])
        if debugging: print "           thetaY = ", y, "\n           thetaZ = ", z
        if y == 0 and z == 0: return 0, ""
        elif y == 0 and z != 0: return z, "z"
        else: return y, "y"
    elif y_or_z == "y":
        i = f.find('stage_tilt_') + 11
        j = f.rfind('thetaY_')
        y = int(f[i:j])
        if debugging: print "           thetaY = ", y
        return y, "y"
    elif y_or_z == "z":
        k = f.rfind('thetaY_') + 7
        l = f.rfind('thetaZ')
        z = int(f[k:l])
        if debugging: print "           thetaZ = ", z
        return z, "z"

def getGainAndACamp(fname):
    i = fname.rfind('cool_G') + 6
    j = fname.find('_', i)
    k = fname.rfind('synth') + 5
    l = fname.find('mV', k)
    gain = float(fname[i:j])
    ACamp = float(fname[k:l])/1000. # Volts
    return gain, ACamp

def getData(fname, get_drive = False, need_time = False, truncate_x = True, use_theta = False, y_or_z = ""):
    """ assumes fname ends with a '.h5' 
        returns unitless data from file """
    if debugging:
        debug_file = fname[fname.rfind('synth'):fname.rfind('.')]
        print "\nDEBUGGING: getting data from ", debug_file
    gain, ACamp = getGainAndACamp(fname)
    if debugging:
        print "           gain of file ", debug_file, " = ", gain
        print "           AC amplitude of file ", debug_file, " = ", ACamp
    f = h5py.File(fname, 'r')
    dset = f['beads/data/pos_data']
    dat = np.transpose(dset) # all this data is in volts
    x = dat[:,bu.xi]
    if (truncate_x and (not use_theta)):
        Fs = dset.attrs['Fsamp']
        half_wavelength = int((Fs/fdrive)/2.)
        if debugging: print '           half_wavelength of ', debug_file, ' is ', half_wavelength
        x = x[:-half_wavelength]
    x_data = ((x - np.average(x))/float(len(x)))/(gain*ACamp) # normalize for calibration
    if debugging: print "           got the x_data of ", debug_file
    if (get_drive or use_theta):
        drive0 = dat[:,bu.drive]
        drive = drive0 - np.average(drive0)
        drive_data = drive/np.std(drive) # normalized, unitless drive
        twice_drive0 = drive0*drive0
        twice_drive = twice_drive0 - np.average(twice_drive0)
        twice_drive_data = twice_drive/np.std(twice_drive)
        if debugging: print "           got the drive_data of ", debug_file, "\n"
        if use_theta:
            position, new_y_or_z = outputThetaPosition(fname, y_or_z)
            return x_data, drive_data, twice_drive_data, position, new_y_or_z
        elif need_time:
            time = dset.attrs['Time']
            return x_data, drive_data, twice_drive_data, time
        else: return x_data, drive_data, twice_drive_data
    else:
        time = dset.attrs['Time']
        if debugging: print ""
        return x_data, time

def getCorrArray(x_data, drive_data, c = 1.):
    """ returns the normalized correlation between the drive and the response """
    corr_array = np.correlate(x_data, drive_data)
    if debugging: print "\nDEBUGGING: getCorrArray() worked! \n"
    return corr_array/c # response in terms of #e at the calibration field

def calibrate(calibration_list, need_drive = True, make_plot = make_calibration_plot, last_plot = False):
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
    phase_array, dat_arrays = ([] for i in range(2))
    if need_drive: drive, twice_drive = ([] for i in range(2))
    if make_plot: t_dat_arrays, tcorr = ([] for i in range(2))
    print "finding phase shift"
    for i in range(N):
        fname = calibration_list[i]
        if debugging: print "           ", i, "th iteration of for loop \n           fname = ", fname
        x_data, drive_data, twice_drive_data = getData(fname, get_drive=True)
        # measure the correlation for normalization purposes
        corr_array = getCorrArray(x_data, drive_data)
        dat_arrays.append(corr_array) # array of all the correlation arrays
        if need_drive:
            # here we average the drive "measurements"
            if drive == []: drive = drive_data/float(N)
            else: drive += drive_data/float(N)

            if twice_drive == []: twice_drive = twice_drive_data/float(N)
            else: twice_drive += twice_drive_data/float(N)
        if make_plot: t_dat_arrays.append(getCorrArray(x_data, twice_drive_data)) # correlation at 2f
        # only work with the first few files to get the phase and c
        if i < 20: phase_array.append(np.argmax(corr_array)) # index of largest correlation coefficient
    index = int(np.average(phase_array))
    print "phase shift is ", index
    corr = []
    for dat_arr in dat_arrays: corr.append(dat_arr[index])
    i = min(len(corr), 20)
    c = np.average(corr[:i])*num_electrons_in_sphere # V^2/electron
    print "calibrating constant c = ", c
    if debugging: print ""
    if make_plot:
        for t_arr in t_dat_arrays: tcorr.append(t_arr[index])
        pc = np.array(corr)/c
        ptc = np.array(tcorr)/c
        x = range(len(corr))
        basic_plot(x, pc, ptc, modifier = 'calibration', last_plot = last_plot)
    if need_drive:
        return index, c, np.array(drive), np.array(twice_drive)
    else:
        return index, c

def correlate(x_data, drive_data, index, c):
    arr = getCorrArray(x_data, drive_data, c = c)
    if debugging: print "\nDEBUGGING: correlate() worked!\n"
    return arr[index]

def getResponseArray(file_list):
    """ assumes file_list is in order """
    x_arr, t_arr = ([] for i in range(2))
    for fname in file_list:
        x, time = getData(fname)
        x_arr.append(x)
        t_arr.append(time)
    t_arr = np.array(t_arr) - t_arr[0]
    if debugging:
        print "\nDEBUGGING: the first element in t_arr is ", t_arr[0]
        print "           getResponseArray() worked!\n"
    return x_arr, t_arr

def corrWithDrive(file_list, calib_list, use_theta = False, last_plot = False, fft = plot_fft):
    x, of, tf = ([] for i in range(3))
    i, c = calibrate(sortFileList(calib_list), need_drive=False)
    if use_theta:
        y_or_z = ""
        i = 0
    if debugging:
        index = min(len(file_list), 20)
        file_list = file_list[:index]
        print "\nDEBUGGING corrWithDrive()\n"
    for f in file_list:
        if use_theta:
            x_data, drive_data, twice_drive_data, position, y_or_z = getData(f, use_theta=True, y_or_z=y_or_z)
            x.append(position)
        else:
            x_data, drive_data, twice_drive_data, time = getData(f, get_drive=True, need_time=True)
            x.append(time)
        of.append(correlate(x_data, drive_data, i, c))
        tf.append(correlate(x_data, twice_drive_data, i, c))
    print "average correlation at drive frequency: ", np.average(of)
    print "average correlation at twice drive frequency: ", np.average(tf)
    if plotting:
        basic_plot(x, of, tf, use_theta=use_theta, last_plot = ((not fft) and last_plot))
        if fft: fft_plot(x, of, last_plot=last_plot)
    if debugging: print ""

def corrWithoutDrive(file_list, drive, twice_drive, index = 0, c = 1., last_plot = False, fft = plot_fft):
    """ takes in a measurement's data array and the phase shift
        returns a plot of the correlation """
    sorted_file_list = sortFileList(file_list)
    if debugging:
        index = min(len(sorted_file_list), 20)
        sorted_file_list = sorted_file_list[:index]
    print "fetching noise data"
    x_arr, t_arr = getResponseArray(sorted_file_list)
    of, tf = ([] for i in range(2))
    if debugging: print "\nDEBUGGING: got response array in plotCorr()"
    for x in x_arr:
        if debugging: print "           iteration in plotCorr number ", len(of)
        of.append(correlate(x, drive, index, c))
        tf.append(correlate(x, twice_drive, index, c))
    print "average correlation at drive frequency: ", np.average(of)
    print "average correlation at twice drive frequency: ", np.average(tf)
    if plotting:
        basic_plot(t_arr, of, tf, last_plot = ((not fft) and last_plot))
        if fft: fft_plot(t_arr, of, last_plot=last_plot)
    if debugging: print ""

def full_correlation_plots(calib_path, data_path, drive_on = False, use_theta = False, last_plot = False, fft = plot_fft):
    """ does everything: calibrates and makes correlation plots at 
        the drive frequency and at twice the drive frequency 
        in total produces 2-4 plots depending on calibration and fft """
    print "\ncalibration from ", calib_path
    print "data from ", data_path
    if debugging: print "\nDEBUGGING: full_correlation_plots"
    calib_list = glob.glob(os.path.join(calib_path, "*.h5"))
    file_list = glob.glob(os.path.join(data_path, "*.h5"))
    if drive_on:
        corrWithDrive(file_list, calib_list, use_theta=use_theta, last_plot=last_plot, fft=fft)
    else:
        i, c, drive, twice_drive = calibrate(calib_list)
        corrWithoutDrive(file_list, drive, twice_drive, i, c, last_plot=last_plot, fft=fft)
    if debugging: print "           plotted f and 2f in full_correlation_plots\n"

""" Now we plot """
if use_as_script:
    ### calibration files
    calib1 = "/data/20170622/bead4_15um_QWP/charge11"
    calib2 = "/data/20170622/bead4_15um_QWP/arb_charge"

    ### this is where the noise files are pulled out
    path1 = "/data/20170622/bead4_15um_QWP/reality_test2"
    path2 = "/data/20170622/bead4_15um_QWP/reality_test3"
    w_path = "/data/20170622/bead4_15um_QWP/dipole27_Y" # this has the W

    full_correlation_plots(calib1, path1, drive_on=True, last_plot=True)

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
