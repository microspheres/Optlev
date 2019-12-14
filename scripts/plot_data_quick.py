import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

refname = r"LPmbar_xyzcool_843.h5"
fname0 = r"LPmbar_xyzcool_842.h5"
path = r"C:\data\20191210\10um\3\newpinhole\acceleration2"

coherence = True

make_plot_vs_time = True
conv_fac = 4.4e-14
if fname0 == "":
	filelist = glob.glob(path+"\*.h5")

	mtime = 0
	mrf = ""
	for fin in filelist:
		f = os.path.join(path, fin) 
		if os.path.getmtime(f)>mtime:
			mrf = f
			mtime = os.path.getmtime(f) 
 
	fname0 = mrf		


		 

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**17

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

        xpsd_outloop, freqs = matplotlib.mlab.psd(dat[:, 4]-numpy.mean(dat[:, 4]), Fs = Fs, NFFT = NFFT)
        # Ddrive = dat[:, bu.drive]*np.gradient(dat[:,bu.drive])
        # DdrivePSD, freqs =  matplotlib.mlab.psd(Ddrive-numpy.mean(Ddrive), Fs = Fs, NFFT = NFFT))
        print pid


        # f, Pxy = sp.csd(dat[:, 0]-numpy.mean(dat[:, 0]), dat[:, 4] - numpy.mean(dat[:, 4]), Fs, nperseg=NFFT)
        # plt.figure()
        # plt.loglog(f, np.sqrt(np.abs(np.real(Pxy))))
	# norm = numpy.median(dat[:, bu.zi])
        if coherence:
                f, Pxy = np.abs(np.real(sp.csd(dat[:, 0]-numpy.mean(dat[:, 0]), dat[:, 4] - numpy.mean(dat[:, 4]), Fs, nperseg=NFFT)))
                f, Pxx = sp.csd(dat[:, 0]-numpy.mean(dat[:, 0]), dat[:, 0]-numpy.mean(dat[:, 0]), Fs, nperseg=NFFT)
                f, Pyy = sp.csd(dat[:, 4]-numpy.mean(dat[:, 4]), dat[:, 4]-numpy.mean(dat[:, 4]), Fs, nperseg=NFFT)
                Cxy = (Pxy**2)/(Pxx*Pyy)
        
	        return [freqs, xpsd, ypsd, dat, zpsd, xpsd_outloop, f, Cxy]
        return [freqs, xpsd, ypsd, dat, zpsd, xpsd_outloop]

data0 = getdata(os.path.join(path, fname0))

def rotate(vec1, vec2, theta):
    vecn1 = numpy.cos(theta)*vec1 + numpy.sin(theta)*vec2
    vecn2 = numpy.sin(theta)*vec1 + numpy.cos(theta)*vec2
    return [vec1, vec2]


if refname:
	data1 = getdata(os.path.join(path, refname))
Fs = 10000
b, a = sp.butter(1, [2*5./Fs, 2*10./Fs], btype = 'bandpass')

if make_plot_vs_time:	

        fig = plt.figure()
        plt.subplot(4, 1, 1)

        plt.plot(data0[3][:,bu.xi] - np.mean(data0[3][:, bu.xi]) )
        if(refname):
                plt.plot(data1[3][:, bu.xi] - np.mean(data1[3][:, bu.xi]) )

        plt.subplot(4, 1, 2)
        plt.plot(data0[3][:, bu.yi] - np.mean(data0[3][:, bu.yi]) )
        if(refname):
                plt.plot(data1[3][:, bu.yi] - np.mean(data1[3][:, bu.yi]) )

        plt.subplot(4, 1, 3)
        plt.plot(data0[3][:, bu.zi] - np.mean(data0[3][:, bu.zi]) )
        if(refname):
                plt.plot(data1[3][:, bu.zi] - np.mean(data1[3][:, bu.zi]) )

        plt.subplot(4, 1, 4)
        plt.plot(data0[3][:, 4] - np.mean(data0[3][:, 4]) )
        if(refname):
                plt.plot(data1[3][:, 4] - np.mean(data1[3][:, 4]) )
       

fig = plt.figure()
plt.subplot(4, 1, 1)
plt.loglog(data0[0], np.sqrt(data0[1]),label="test")
if refname:
	plt.loglog(data1[0], np.sqrt(data1[1]),label="ref")
plt.ylabel("V/rtHz")
plt.legend(loc=3)
plt.subplot(4, 1, 2)
plt.loglog(data0[0], np.sqrt(data0[2]))
if refname:
	plt.loglog(data1[0], np.sqrt(data1[2]))
plt.subplot(4, 1, 3)
plt.loglog(data0[0],  np.sqrt(data0[4]))
if refname:
	plt.loglog(data1[0], np.sqrt(data1[4]))
plt.ylabel("V/rtHz")
plt.xlabel("Frequency[Hz]")

plt.subplot(4, 1, 4)
plt.loglog(data0[0],  np.sqrt(data0[5]))
if refname:
	plt.loglog(data1[0], np.sqrt(data1[5]) )
plt.ylabel("V/rtHz")
plt.xlabel("Frequency[Hz]")


if refname and coherence:
        plt.figure()
        plt.plot(data0[6], np.sqrt(data0[7]))
        plt.plot(data1[6], np.sqrt(data1[7]))
        plt.grid()
        plt.ylim(0., 1.01)
        plt.xlim(1, 120)

plt.show()

# fig = plt.figure
# plt.plot(data0[0], np.sqrt(data0[1]))
# plt.show()
