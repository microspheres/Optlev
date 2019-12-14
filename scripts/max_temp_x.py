import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt

LT = r"1mbar_zcool.h5"
HT = r"1mbar_yzcool_xheat18.h5"
path = r"C:\data\paper3\22um\PreChamber_ATM\max_temp_x\1mbar"

T0 = 1000. #kelvin	 

NFFT = 2**17

freq_fit_min = 60.
freq_fit_max = 110.

def getdata(fname):
	print "Opening file: ", fname
	## guess at file type from extension
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		#max_volt = dset.attrs['max_volt']
		#nbit = dset.attrs['nbit']
		Fs = dset.attrs['Fsamp']
                pid = dset.attrs['PID']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                Press = dset.attrs['pressures'][0]
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT) 
	ypsd, freqs = matplotlib.mlab.psd(dat[:, bu.yi]-numpy.mean(dat[:, bu.yi]), Fs = Fs, NFFT = NFFT)
        zpsd, freqs = matplotlib.mlab.psd(dat[:, bu.zi]-numpy.mean(dat[:, bu.zi]), Fs = Fs, NFFT = NFFT)
        # Ddrive = dat[:, bu.drive]*np.gradient(dat[:,bu.drive])
        # DdrivePSD, freqs =  matplotlib.mlab.psd(Ddrive-numpy.mean(Ddrive), Fs = Fs, NFFT = NFFT))
        print pid
        print Press
	norm = numpy.median(dat[:, bu.zi])

        # plt.figure()
        # plt.plot(dat[:, bu.xi])
        # plt.show()
        
	return [freqs, xpsd, ypsd, dat, zpsd]

def psd2(f, A, f0, gamma):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    gamma1 = 2.0*np.pi*gamma
    s1 = 2.*(gamma1*(w0**2))
    s2 = 1.*(w0**2)*((w0**2 - w**2)**2 + (gamma1*w)**2)
    s = np.sqrt(s1/s2)
    return A*(s)**2

def area_ratio(fileLT, fileHT, t0):
    a = getdata(fileLT)
    freq = a[0]
    xpsd2 = a[1]

    index0 = np.where( freq >= freq_fit_min )[0][0]
    index1 = np.where( freq >= freq_fit_max )[0][0]

    fit_points1 = np.logical_and(freq > freq_fit_min, freq < 59.)
    fit_points2 = np.logical_and(freq > 61, freq < freq_fit_max)

    fit_points = fit_points1 + fit_points2
    
    poptLT, pcovLT = opt.curve_fit(psd2, freq[fit_points], xpsd2[fit_points])

    b = getdata(fileHT)
    xpsd2HT = b[1]

    fres = poptLT[1]
    print fres
    p0 = np.array([3500000, 82.2, 0.00008])
    poptHT, pcovHT = opt.curve_fit(psd2, freq[fit_points], xpsd2HT[fit_points], p0 = p0)

    areaLT = np.sum(xpsd2[fit_points])
    areaHT = np.sum(xpsd2HT[fit_points])
    ratio = 1.0*areaHT/areaLT
    
    return ratio*t0


    

# plt.figure()
# plt.loglog(freq, xpsd2)
# plt.loglog(freq, psd2(freq, *poptLT))
# plt.loglog(freq, xpsd2HT)
# plt.loglog(freq, psd2(freq, *poptHT))
# plt.show()


fileLT = os.path.join(path, LT)
fileHT = os.path.join(path, HT)
print area_ratio(fileLT, fileHT, T0)
