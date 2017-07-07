import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import numpy, h5py, matplotlib, os, glob
from bead_util import get_color_map
import bead_util as bu

### The functions "saveACfile" and "saveACandDCfile" take in a path string (as below)
### then find the file_list and save the values from that list
### so you don't actually have to do anything other than type in "saveACfile(path)".

path = '/data/20170511/bead2_15um_QWP/new_sensor_feedback/charge43_whole_points/60.0_74.9_75.4'
conversion = 3.7139625927e-13 # N/V
Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2 ** 17 # number of bins
#file_list = glob.glob(path+"/*.h5")
#integrationTime = 20 # seconds

# for now pretend our integration time was 100s because that's what it will eventually be
integrationTime = 100

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
    xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi] - numpy.mean(dat[:, bu.xi]), Fs=Fs, NFFT=NFFT)
    drive, freqs = matplotlib.mlab.psd(dat[:, bu.drive] - numpy.mean(dat[:, bu.drive]), Fs=Fs, NFFT=NFFT)
    freqs = np.array(freqs)
    xpsd = np.array(xpsd)
    drive = np.array(drive)
    return freqs, xpsd, drive

def getACAmplitudeGraphs(file_list, make_plots = False, zeroDC = True):
    """output AC voltages and corresponding amplitudes at both omega and 2 omega for a DC voltage of 0 or not zero"""
    N = len(file_list)
    ax = {} # input only numpy arrays as values; this is the SQUARE of the desired PSD
    adx = {} # input only numpy arrays as values; SQUARE of the desired drive PSD
    voltageCount = {} # input integers that count how many times an AC voltage value has shown up
    setDC = 0
    for index in range(N):
        f = file_list[index]
        freqs, xpsd, drive = getdata(f)
        i = f.rfind("synth")+5
        j = f.rfind("mV")
        k = f.rfind("mV",0,j)
        l = f.rfind("Hz") + 2
        ACvoltage = float(f[i:k])/1000. # V
        DCvoltage = float(f[l:j])/1000. # V
        if ACvoltage in ax:
            if zeroDC:
                if DCvoltage == 0:
                    voltageCount[ACvoltage] += 1
                    ax[ACvoltage] += xpsd
                    adx[ACvoltage] += drive
            else:
                if DCvoltage != 0:
                    setDC = DCvoltage
                    voltageCount[ACvoltage] += 1
                    ax[ACvoltage] += xpsd
                    adx[ACvoltage] += drive
        else:
            voltageCount[ACvoltage] = 1
            ax[ACvoltage] = xpsd
            adx[ACvoltage] = drive
    binF = freqs[2] - freqs[1]
    # print 'binF is ' + str(binF) # gave 0.0762939453125
    ACvoltages = sorted(ax.keys())
    N1 = len(ACvoltages)
    keyPicked = np.amax(ACvoltages)
    dxPicked = np.sqrt(adx[keyPicked])
    indexPicked = np.argmax(dxPicked)
    DCvoltages = [setDC] * N1
    omegaAmplitudes = np.arange(N1)
    twoOmegaAmplitudes = np.arange(N1)
    if make_plots:
        psd_plots = range(N1)
        #drive_plots = range(N1)
    """Now insert the amplitude for the requisite frequencies"""
    for index in range(N1):
        volt = ACvoltages[index] # V
        constant = conversion # N/V
        i = indexPicked
        psd = np.sqrt(ax[volt]/voltageCount[volt]) # V/sqrtHz
        if make_plots:
            psd_plots[index] = constant*psd/np.sqrt(integrationTime) # so this is in N
            #drive_plots[index] = np.sqrt(adx[volt])
        omegaAmplitudes[index] = constant*psd[i]*np.sqrt(binF) # N
        twoOmegaAmplitudes[index] = constant*psd[2*i+1]*np.sqrt(binF) # N
    if make_plots:
        plot_psds(psd_plots, freqs, ACvoltages, indexPicked)
    return ACvoltages, omegaAmplitudes, twoOmegaAmplitudes, DCvoltages

def plot_psds(psd_plots, frequencies, labels, index):
    colorList = get_color_map(len(psd_plots))
    plt.figure()
    for currLabel, psd, color in zip(labels, psd_plots, colorList):
        plt.loglog(frequencies[index], psd[index], "x", color = color)
        plt.loglog(frequencies[2*index+1], psd[2*index+1], "x", color = color)
        #plt.loglog(frequencies, drive, color = color)
        plt.loglog(frequencies, psd, color = color, label = currLabel)
    plt.title("Noise level integrated over "+str(integrationTime)+" seconds")
    plt.xlabel("Frequencies [Hz]")
    plt.xlim([20,100])
    plt.ylabel("Noise Level [N]")
    plt.ylim([0, np.amax(psd_plots[-1][np.argmin(np.abs(frequencies - 20)):np.argmin(np.abs(frequencies - 100))])])
    #plt.title(path[path.rfind('\\'):])
    plt.legend()
    plt.show(block = False)

# Make the plots requested here
#file_list = glob.glob(path+"/*.h5")
#getACAmplitudeGraphs(file_list, make_plots = True, zeroDC = True)

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


# ONLY USE THESE COMMANDS ON THE WINDOWS COMPUTER WHEN WRITING INTO DATA
#saveACfile(path)
#saveACandDCfile(path)
