import numpy as np
import matplotlib.pyplot as plt
import os, glob, h5py
import bead_util as bu

# this is how many of the calibration files will be used to scale the correlation values
num_first_files = 20

### calibration files
path = "/data/20170622/bead4_15um_QWP/charge9"
file_list = glob.glob(os.path.join(path, "*.h5"))

### this is where the noise files are pulled out
path1 = "/data/20170622/bead4_15um_QWP/reality_test"
file_list1 = glob.glob(os.path.join(path1,"*.h5"))
path2 = "/data/20170622/bead4_15um_QWP/reality_test2"
file_list2 = glob.glob(os.path.join(path2,"*.h5"))

fdrive = 41. # Hz
#wavelength = Fs/fdrive because Fs = samples/second

def sortFileList(file_list):
    N = len(file_list)
    new_list = ['']*N
    for i in range(N):
        f = file_list[i]
        j = f.rfind('_') + 1
        k = f.rfind('.')
        n = int(f[j:k])
        new_list[n] = f
    return new_list

def getData(fname):
    """ assuming fname ends with a '.h5' 
        returns an array of a truncated x array and the drive array """
    f = h5py.File(fname, 'r') # read the file
    dset = f['beads/data/pos_data']
    Fs = float(dset.attrs['Fsamp'])
    half_wavelength = int((Fs/fdrive)/2.)
    dat = np.transpose(dset)
    dat = dat * 10. / (2. ** 15. - 1.)
    x = dat[:,bu.xi]
    x_data = x[:-half_wavelength]
    drive_data = dat[:,bu.drive]
    return [x_data, drive_data]

def getCorrArray(fdat, c = 1):
        # we have to divide out the drive voltage but that needs to be calculated from the drive data
        # as we don't have that right now, we will temporarily be assuming a sine wave
        # so we take the standard deviation of the drive and multiply by sqrt(2)
    x_data = fdat[0] - np.average(fdat[0]) # Volts of response
    drive = fdat[1] - np.average(fdat[1]) # Volts of drive
    damp = np.std(drive)*np.sqrt(2.) # Volts of drive
    #print 'drive amplitude is ',damp
    raw_corr_array = np.correlate(x_data, drive)/len(x_data) # V^2
    return raw_corr_array/(damp*c) # response in terms of e at the calibration field

def calibrate(calibration_data_list):
    """ goes through the x and drive data of each file (inputs)
        returns the index of the phase shift (independant of normalization)
                and the normalization value of one electron
                (assuming calibration_data is of only one electron)
        using only the first num_first_files files
        assume calibration_data_list is sorted"""
    calib_data = calibration_data_list[:num_first_files]
    phase_array = []
    dat_arrays = []
    for fdat in calib_data:
        corr_array = getCorrArray(fdat)
        dat_arrays.append(corr_array)
        phase_array.append(np.argmax(corr_array))
    index = int(np.average(phase_array))
    c_approx = []
    for dat_arr in dat_arrays:
        c_approx.append(dat_arr[index])
    c = np.average(c_approx)
    return index, c

def correlate(fdat, index, c):
    arr = getCorrArray(fdat, c = c)
    return arr[index]

def getDataArray(file_list):
    data_arr = []
    for fname in file_list:
        data_arr.append(getData(fname))
    return data_arr

def plotCorr(data_arr, index, c, calibration = False):
    """ takes in a measurement's data array and the phase shift
        returns a plot of the correlation """
    y = []
    for data in data_arr:
        y.append(correlate(data, index, c))
    plt.figure()
    plt.plot(y)
    if calibration:
        plt.title('Calibration data correlation')
        b = True
    else:
        plt.title('Measurement correlation')
        b = False
    plt.show(block = b)


""" Now we plot """
# first sort the file lists
calib_list = sortFileList(file_list)
#print calib_list
flist1 = sortFileList(file_list1)
flist2 = sortFileList(file_list2)

# get the data
print 'fetching calibration data'
calibDat = getDataArray(calib_list)
print 'fetching noise data'
n = len(flist1)/10
dat1 = getDataArray(flist1[:n])
dat2 = getDataArray(flist2[:n])

# find the phase shift
print 'finding phase shift'
index, c = calibrate(calibDat)
print 'phase shift: ', index

# plot the two measured noise data
print 'plotting noise data'
plotCorr(dat1, index, c)
plotCorr(dat2, index, c)

# plot the calibration data
print 'plotting calibration data'
plotCorr(calibDat, index, c, calibration = True)

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
