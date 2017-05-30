import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
from scipy.optimize import curve_fit

# refname = r"4mbar_nocool.h5"
#fname0 = r"15um_1mbar_nocool_G200.h5"
fname0 = r"5umPowerSpectrum.h5"

path = r"C:\Users\Sumita\Documents\Microspheres\5 and 15 micron test data"

make_plot_vs_time = True
conv_fac = 4.4e-14
if fname0 == "":
    filelist = os.listdir(path)

    mtime = 0
    mrf = ""
    for fin in filelist:
        f = os.path.join(path, fin)
        if os.path.getmtime(f) > mtime:
            mrf = f
            mtime = os.path.getmtime(f)

    fname0 = mrf

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2 ** 17


def getdata(fname):
    print "Opening file: ", fname
    ## guess at file type from extension
    _, fext = os.path.splitext(fname)
    if (fext == ".h5"):
        f = h5py.File(fname, 'r')
        dset = f['beads/data/pos_data']
        dat = numpy.transpose(dset)
        # max_volt = dset.attrs['max_volt']
        # nbit = dset.attrs['nbit']
        Fs = dset.attrs['Fsamp']

        # dat = 1.0*dat*max_volt/nbit
        dat = dat * 10. / (2 ** 15 - 1)

    else:
        dat = numpy.loadtxt(fname, skiprows=5, usecols=[2, 3, 4, 5, 6])

    xpsd, freqs = matplotlib.mlab.psd(dat[:, 0] - numpy.mean(dat[:, 0]), Fs=Fs, NFFT=NFFT)
    ypsd, freqs = matplotlib.mlab.psd(dat[:, 1] - numpy.mean(dat[:, 1]), Fs=Fs, NFFT=NFFT)
    zpsd, freqs = matplotlib.mlab.psd(dat[:, 2] - numpy.mean(dat[:, 2]), Fs=Fs, NFFT=NFFT)

    norm = numpy.median(dat[:, 2])
    # for h in [xpsd, ypsd, zpsd]:
    #        h /= numpy.median(dat[:,2])**2
    return [freqs, xpsd, ypsd, dat, zpsd]


data0 = getdata(os.path.join(path, fname0))

# def rotate(vec1, vec2, theta):
#    vecn1 = numpy.cos(theta)*vec1 + numpy.sin(theta)*vec2
#    vecn2 = numpy.sin(theta)*vec1 + numpy.cos(theta)*vec2
#    return [vec1, vec2]

# if refname:
#	data1 = getdata(os.path.join(path, refname))
# Fs = 10000
# b, a = sp.butter(1, [2*5./Fs, 2*10./Fs], btype = 'bandpass')


Pi = np.pi
T = 300
Kb = 1.38 * (1e-23)
# omega = 190
M = 2100 * ((2.5e-6) ** 3) * (4. / 3) * Pi


# def PSD(w, damp, Nw,C):
#    return C*((2.0*Kb*T)/M)*(damp/((Nw**2 - w**2)**2 + (w*damp)**2))

def PSD2(w, a, W, damp):
    return a * (damp / ((W ** 2 - (w) ** 2) ** 2 + (w * damp) ** 2))


xdata = 2 * Pi * data0[0][10:3000]
ydata = data0[4][10:3000]

# xdata = 2*Pi*data0[0][50:8000]
# ydata = data0[1][50:8000]


popt, pcov = curve_fit(PSD2, xdata, ydata)

plt.loglog(xdata, PSD2(xdata, *popt), 'g--', label='')
plt.loglog(xdata, ydata, label="data")

Conv = popt[0] / ((2 * Kb * T) / M)

print Conv
print popt
