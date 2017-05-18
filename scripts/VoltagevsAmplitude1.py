import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import numpy, h5py, matplotlib, os, glob
from bead_util import get_color_map

### The functions "saveACfile" and "saveACandDCfile" take in a path string (as below)
### then find the file_list and save the values from that list
### so you don't actually have to do anything other than type in "saveACfile(path)".

path = r"C:\data\20170511\bead2_15um_QWP\new_sensor_feedback\charge45_whole_points\60.0_74.9_0.0"
conversion = 4.1e-13
Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2 ** 17
#file_list = glob.glob(path+"/*.h5")

def getdata(fname):
    ## guess at file type from extension
    _, fext = os.path.splitext(fname)
    if (fext == ".h5"):
        f = h5py.File(fname, 'r')
        dset = f['beads/data/pos_data']
        dat = numpy.transpose(dset)
        Fs = dset.attrs['Fsamp']
        dat = dat * 10. / (2 ** 15 - 1)
    else:
        dat = numpy.loadtxt(fname, skiprows=5, usecols=[2, 3, 4, 5, 6])
    xpsd, freqs = matplotlib.mlab.psd(dat[:, 0] - numpy.mean(dat[:, 0]), Fs=Fs, NFFT=NFFT)
    drive, freqs = matplotlib.mlab.psd(dat[:, 5] - numpy.mean(dat[:, 5]), Fs=Fs, NFFT=NFFT)
    return [freqs, xpsd, drive]

def getACAmplitudeGraphs(file_list, make_plots = False, zeroDC = True):
    """output AC voltages and corresponding amplitudes at both omega and 2 omega for a DC voltage of 0 or not zero"""
    N = len(file_list)
    x = {} # input only numpy arrays as values
    dx = {} # input only numpy arrays as values
    voltageCount = {} # input integers that count how many times an AC voltage value has shown up
    for index in range(N):
        f = file_list[index]
        a = getdata(f)
        i = f.rfind("synth")+5
        j = f.rfind("mV")
        k = f.rfind("mV",0,j)
        l = f.rfind("Hz") + 2
        ACvoltage = float(f[i:k])/1000.
        DCvoltage = float(f[l:j])/1000.
        if ACvoltage in x:
            if zeroDC:
                if DCvoltage == 0:
                    voltageCount[ACvoltage] += 1
                    x[ACvoltage] += numpy.sqrt(a[1])
                    dx[ACvoltage] += numpy.sqrt(a[2])
            else:
                if DCvoltage != 0:
                    voltageCount[ACvoltage] += 1
                    x[ACvoltage] += numpy.sqrt(a[1])
                    dx[ACvoltage] += numpy.sqrt(a[2])
        else:
            voltageCount[ACvoltage] = 1
            x[ACvoltage] = numpy.sqrt(a[1])
            dx[ACvoltage] = numpy.sqrt(a[2])
    ACvoltages = sorted(x.keys())
    N1 = len(ACvoltages)
    keyPicked = np.amax(ACvoltages)
    dxPicked = dx[keyPicked]
    indexPicked = np.argmax(dxPicked)
    DCvoltages = [0] * N1
    omegaAmplitudes = range(N1)
    twoOmegaAmplitudes = range(N1)
    if make_plots:
        psd_plots = range(N1)
        #drive_plots = range(N1)
    """Now insert the amplitude for the requisite frequencies"""
    for index in range(N1):
        volt = ACvoltages[index]
        constant = conversion/voltageCount[volt]
        #i = numpy.argmax(dx[volt])
        i = indexPicked
        psd = x[volt]
        if make_plots:
            psd_plots[index] = constant*psd
            #drive_plots[index] = dx[volt]
        omegaAmplitudes[index] = constant*psd[i]
        twoOmegaAmplitudes[index] = constant*psd[2*i]
    if make_plots:
        plot_psds(psd_plots, a[0], ACvoltages, indexPicked)
    return ACvoltages, omegaAmplitudes, twoOmegaAmplitudes, DCvoltages

def plot_psds(psd_plots, frequencies, labels, index):
    colorList = get_color_map(len(psd_plots))
    plt.figure()
    for currLabel, psd, color in zip(labels, psd_plots, colorList):
        plt.plot(frequencies[index], psd[index], "x", color = color)
        plt.plot(frequencies[2*index+1], psd[2*index+1], "x", color = color)
        #plt.plot(frequencies, drive, color = color)
        plt.plot(frequencies, psd, color = color, label = currLabel)
    plt.xlabel("Frequencies [Hz]")
    plt.xlim([20,100])
    plt.ylabel("Intensity [N/sqrt(Hz)]")
    plt.ylim([0, np.amax(psd_plots[-1][np.argmin(np.abs(frequencies - 20)):np.argmin(np.abs(frequencies - 100))])])
    plt.title(path[path.rfind('\\'):])
    plt.legend()
    plt.show()

# getACAmplitudeGraphs(file_list, make_plots = False, zeroDC = True)

def saveACfile(path, make_plots = False):
    file_list = glob.glob(path+"/*.h5")
    ACvoltages, omegaAmplitudes, twoOmegaAmplitudes, DCvoltages = getACAmplitudeGraphs(file_list, make_plots)
    np.savetxt(path+'/ACamplitudes.txt', (ACvoltages, omegaAmplitudes, twoOmegaAmplitudes, DCvoltages))
    return

def saveACandDCfile(path, make_plots = False):
    file_list = glob.glob(path+"/*.h5")
    ACvoltages, omegaAmplitudes, twoOmegaAmplitudes, DCvoltages = getACAmplitudeGraphs(file_list, make_plots, zeroDC = False)
    np.savetxt(path+'/ACandDCamplitudes.txt', (ACvoltages, omegaAmplitudes, twoOmegaAmplitudes, DCvoltages))
    return



saveACfile(path)

saveACandDCfile(path)