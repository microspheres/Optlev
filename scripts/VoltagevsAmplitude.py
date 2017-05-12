import numpy, h5py, matplotlib, os, glob
path = r"/data/20170511/bead2_15um_QWP/new_sensor_feedback/charge7_piezo_56.9_74.9_75.4"
conversion = 4.1e-13
Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2 ** 13
file_list = glob.glob(path+"*.h5")

def getdata(fname):
    print "Opening file: ", fname
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

def getACAmplitudeGraphs(file_list):
    """output AC voltages and corresponding amplitudes at both omega and 2 omega"""
    N = len(file_list)
    constant = conversion/N
    x = {} # input only numpy arrays as values
    dx = {} # input only numpy arrays as values
    voltageCount = {} # input integers that count how many times an AC voltage value has shown up
    for index in range(N):
        f = file_list[index]
        a = getdata(f)
        i = f.rfind("synth")+5
        j = f.rfind("mV")
        k = f.rfind("mV",0,j)
        ACvoltage = float(f[i:k])/1000.
        if ACvoltage in x:
            voltageCount[ACvoltage] += 1
            x[ACvoltage] += numpy.sqrt(a[1])
            dx[ACvoltage] += numpy.sqrt(a[2])
        else:
            voltageCount[ACvoltage] = 1
            x[ACvoltage] = numpy.sqrt(a[1])
            dx[ACvoltage] = numpy.sqrt(a[2])
    ACvoltages = x.keys()
    N1 = len(ACvoltages)
    omegaAmplitudes = range(N1)
    twoOmegaAmplitudes = range(N1)
    """Now insert the amplitude for the requisite frequencies"""
    for index in range(N1):
        i = numpy.argmax(dx[ACvoltages[index]])
        omegaAmplitudes[index] = constant*x[i]
        twoOmegaAmplitudes[index] = constant*x[2*i]
    return ACvoltages, omegaAmplitudes, twoOmegaAmplitudes

ACvoltages, omegaAmplitudes, twoOmegaAmplitudes = getACAmplitudeGraphs(file_list)

np.savetxt(path+'/amplitudes.txt', (ACvoltages, omegaAmplitudes, twoOmegaAmplitudes))
