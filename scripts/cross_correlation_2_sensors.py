import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu

refname = r"test2.h5"
fname0 = r"15um_QWP\1mbar_500s_G10.h5"
path = r"C:\data\20170523_cross_correlation"
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

def getdata_cross(fname):
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

	xcsd, freqs = matplotlib.mlab.csd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), dat[:, bu.xl]-numpy.mean(dat[:, bu.xl]), Fs = Fs, NFFT = NFFT) 
	ycsd, freqs = matplotlib.mlab.csd(dat[:, bu.yi]-numpy.mean(dat[:, bu.yi]), dat[:, bu.yl]-numpy.mean(dat[:, bu.yl]), Fs = Fs, NFFT = NFFT)
        zcsd, freqs = matplotlib.mlab.csd(dat[:, bu.zi]-numpy.mean(dat[:, bu.zi]), dat[:, bu.zl]-numpy.mean(dat[:, bu.zl]), Fs = Fs, NFFT = NFFT)
        xcsd_old, freqs = matplotlib.mlab.csd(dat[:, bu.xi_old]-numpy.mean(dat[:, bu.xi_old]), dat[:, bu.xl]-numpy.mean(dat[:, bu.xl]), Fs = Fs, NFFT = NFFT)
        # Ddrive = dat[:, bu.drive]*np.gradient(dat[:,bu.drive])
        # DdrivePSD, freqs =  matplotlib.mlab.csd(Ddrive-numpy.mean(Ddrive), Fs = Fs, NFFT = NFFT)


	norm = numpy.median(dat[:, bu.zi])
        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,bu.zi])**2
	return [freqs, xcsd, ycsd, dat, zcsd, xcsd_old]


def getdata_cohere(fname):
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

	xcsd, freqs = matplotlib.mlab.cohere(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), dat[:, bu.xl]-numpy.mean(dat[:, bu.xl]), Fs = Fs, NFFT = NFFT) 
	ycsd, freqs = matplotlib.mlab.cohere(dat[:, bu.yi]-numpy.mean(dat[:, bu.yi]), dat[:, bu.yl]-numpy.mean(dat[:, bu.yl]), Fs = Fs, NFFT = NFFT)
        zcsd, freqs = matplotlib.mlab.cohere(dat[:, bu.zi]-numpy.mean(dat[:, bu.zi]), dat[:, bu.zl]-numpy.mean(dat[:, bu.zl]), Fs = Fs, NFFT = NFFT)
        xcsd_old, freqs = matplotlib.mlab.cohere(dat[:, bu.xi_old]-numpy.mean(dat[:, bu.xi_old]), dat[:, bu.xl]-numpy.mean(dat[:, bu.xl]), Fs = Fs, NFFT = NFFT)
        # Ddrive = dat[:, bu.drive]*np.gradient(dat[:,bu.drive])
        # DdrivePSD, freqs =  matplotlib.mlab.csd(Ddrive-numpy.mean(Ddrive), Fs = Fs, NFFT = NFFT)


	norm = numpy.median(dat[:, bu.zi])
        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,bu.zi])**2
	return [freqs, xcsd, ycsd, dat, zcsd, xcsd_old]




data0 = getdata_cohere(os.path.join(path, fname0))

# def rotate(vec1, vec2, theta):
#     vecn1 = numpy.cos(theta)*vec1 + numpy.sin(theta)*vec2
#     vecn2 = numpy.sin(theta)*vec1 + numpy.cos(theta)*vec2
#     return [vec1, vec2]


if refname:
	data1 = getdata_cohere(os.path.join(path, refname))
Fs = 10000
b, a = sp.butter(1, [2*5./Fs, 2*10./Fs], btype = 'bandpass')

if make_plot_vs_time:	

        fig = plt.figure()
        plt.subplot(3, 1, 1)

        plt.plot(data0[3][:,0] - np.mean(data0[3][:, 0]) )
        if(refname):
                plt.plot(data1[3][:, bu.xi] - np.mean(data1[3][:, bu.xi]) )

        plt.subplot(3, 1, 2)
        plt.plot(data0[3][:, bu.yi] - np.mean(data0[3][:, bu.yi]) )
        if(refname):
                plt.plot(data1[3][:, bu.yi] - np.mean(data1[3][:, bu.yi]) )

        plt.subplot(3, 1, 3)
        plt.plot(data0[3][:, bu.zi] - np.mean(data0[3][:, bu.zi]) )
        if(refname):
                plt.plot(data1[3][:, bu.zi] - np.mean(data1[3][:, bu.zi]) )
       

fig = plt.figure()
plt.subplot(4, 1, 1)
plt.loglog(data0[0], np.sqrt(data0[1]),label="test")
if refname:
	plt.loglog(data1[0], np.sqrt(data1[1]),label="ref")
plt.ylabel("Coherence")
plt.legend(loc=3)
plt.subplot(4, 1, 2)
plt.loglog(data0[0], np.sqrt(data0[2]))
if refname:
	plt.loglog(data1[0], np.sqrt(data1[2]))
plt.subplot(4, 1, 3)
plt.loglog(data0[0],  np.sqrt(data0[4]))
if refname:
	plt.loglog(data1[0], np.sqrt(data1[4]))
plt.ylabel("Coherence")
plt.subplot(4, 1, 4)
plt.loglog(data0[0],  np.sqrt(data0[5]))
if refname:
        plt.loglog(data1[0], np.sqrt(data1[5]))
plt.xlabel("Frequency[Hz]")
plt.show()

# fig = plt.figure
# plt.loglog(data1[0], np.sqrt(data1[6]))
# plt.show()
