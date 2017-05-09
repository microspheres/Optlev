import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np


refname = r"1mbar_zcool_G5.h5"
fname0 = r""
path = r"C:\data\20170504\bead9_15um_QWP\new_sensor_feedback"
# refname = r"C:\data\20170403\bead6_15um"
# fname0 = r"xout_100Hz_1.h5"
# path = r"C:\Data\20170224\xy_test\feedback_test"
make_plot_vs_time = True
conv_fac = 4.4e-14
if fname0 == "":
	filelist = os.listdir(path)

	mtime = 0
	mrf = ""
	for fin in filelist:
		f = os.path.join(path, fin) 
		if os.path.getmtime(f)>mtime:
			mrf = f
			mtime = os.path.getmtime(f) 
 
	fname0 = mrf		


		 

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**12

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
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, 0]-numpy.mean(dat[:, 0]), Fs = Fs, NFFT = NFFT) 
	ypsd, freqs = matplotlib.mlab.psd(dat[:, 1]-numpy.mean(dat[:, 1]), Fs = Fs, NFFT = NFFT)
        zpsd, freqs = matplotlib.mlab.psd(dat[:, 2]-numpy.mean(dat[:, 2]), Fs = Fs, NFFT = NFFT)
        xpsd_new, freqs = matplotlib.mlab.psd(dat[:, 6]-numpy.mean(dat[:, 6]), Fs = Fs, NFFT = NFFT)


	norm = numpy.median(dat[:, 2])
        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,2])**2
	return [freqs, xpsd, ypsd, dat, zpsd, xpsd_new]

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
        plt.subplot(3, 1, 1)

        plt.plot(data0[3][:,0] - np.mean(data0[3][:, 0]) )
        if(refname):
                plt.plot(data1[3][:, 0] - np.mean(data1[3][:, 0]) )

        plt.subplot(3, 1, 2)
        plt.plot(data0[3][:, 1] - np.mean(data0[3][:, 1]) )
        if(refname):
                plt.plot(data1[3][:, 1] - np.mean(data1[3][:, 1]) )

        plt.subplot(3, 1, 3)
        plt.plot(data0[3][:, 2] - np.mean(data0[3][:, 2]) )
        if(refname):
                plt.plot(data1[3][:, 2] - np.mean(data1[3][:, 2]) )
       

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
plt.subplot(4, 1, 4)
plt.loglog(data0[0],  np.sqrt(data0[5]))
if refname:
        plt.loglog(data1[0], np.sqrt(data1[5]))
plt.xlabel("Frequency[Hz]")
plt.show()
