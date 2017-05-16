import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import os

conv_fac = 4.4e-14
Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2 ** 12

def getdata(fname):
    print "Opening file: ", fname
    ## guess at file type from extension
    _, fext = os.path.splitext(fname)
    if (fext == ".h5"):
        f = h5py.File(fname, 'r')
        dset = f['beads/data/pos_data']
        dat = np.transpose(dset)

        Fs = dset.attrs['Fsamp']

        dat = dat * 10. / (2 ** 15 - 1)

    else:
        dat = np.loadtxt(fname, skiprows=5, usecols=[2, 3, 4, 5, 6])

    xpsd, freqs = matplotlib.mlab.psd(dat[:, 0] - np.mean(dat[:, 0]), Fs=Fs, NFFT=NFFT)
    ypsd, freqs = matplotlib.mlab.psd(dat[:, 1] - np.mean(dat[:, 1]), Fs=Fs, NFFT=NFFT)
    zpsd, freqs = matplotlib.mlab.psd(dat[:, 2] - np.mean(dat[:, 2]), Fs=Fs, NFFT=NFFT)

    return [freqs, xpsd, ypsd, dat, zpsd]


def formatData(rfile1, rfile2, path):
    """both rfile1 and rfile2 come in the following form"""
    # rfile1,2 = r"filename.h5"
    # path = r"C:\Users\Sumita\Documents\Microspheres\5 and 15 micron test data\15 um"

    if rfile2 == "":
        filelist = os.listdir(path)

        mtime = 0
        mrf = ""
        for fin in filelist:
            f = os.path.join(path, fin)
            if os.path.getmtime(f) > mtime:
                mrf = f
                mtime = os.path.getmtime(f)

        rfile2 = mrf

    data0 = getdata(os.path.join(path, rfile2))
    data1 = getdata(os.path.join(path, rfile1))
    return data0, data1


def makePlotVsTime(rfile1, rfile2, path):
    data0, data1 = formatData(rfile1, rfile2, path)

    plt.figure()
    plt.subplot(3, 1, 1)

    plt.plot(data0[3][:, 0] - np.mean(data0[3][:, 0]))
    if (data1):
        plt.plot(data1[3][:, 0] - np.mean(data1[3][:, 0]))

    plt.subplot(3, 1, 2)
    plt.plot(data0[3][:, 1] - np.mean(data0[3][:, 1]))
    if (data1):
        plt.plot(data1[3][:, 1] - np.mean(data1[3][:, 1]))

    plt.subplot(3, 1, 3)
    plt.plot(data0[3][:, 2] - np.mean(data0[3][:, 2]))
    if (data1):
        plt.plot(data1[3][:, 2] - np.mean(data1[3][:, 2]))
    plt.show()


def plotData(rfile1, rfile2, path):
    data0, data1 = formatData(rfile1, rfile2, path)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.loglog(data0[0], np.sqrt(data0[1]), label="File 2")
    if rfile1:
        plt.loglog(data1[0], np.sqrt(data1[1]), label="File 1")
    plt.legend(loc=3)
    plt.subplot(3, 1, 2)
    plt.loglog(data0[0], np.sqrt(data0[2]))
    if rfile1:
        plt.loglog(data1[0], np.sqrt(data1[2]))
    plt.ylabel("V/rt Hz")
    plt.subplot(3, 1, 3)
    plt.loglog(data0[0], np.sqrt(data0[4]))
    if rfile1:
        plt.loglog(data1[0], np.sqrt(data1[4]))
    plt.xlabel("Frequency [Hz]")
    plt.show()


def plotOne(rfile, path, name):
    """ rfile1 = r"filename.h5"
        path = r"C:\Users\Sumita\Documents\Microspheres\5 and 15 micron test data" """
    """plt.figure()
       plotOne(file1, path1, name1)
       plotOne(file2, path2, name2)
       plt.title("title")
       plt.show()"""
    data = getdata(os.path.join(path, rfile))
    plt.loglog(data[0], np.sqrt(data[4]), label = name)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("V/rt Hz")
