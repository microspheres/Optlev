import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu


refname = r"1mbar_G5_zcool.h5"
fname0 = r"1mbar_G5_zcool.h5"
path = r"C:\data\20170526\bead4_15um_QWP"
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
NFFT = 2**13

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

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT) 
	ypsd, freqs = matplotlib.mlab.psd(dat[:, bu.yi]-numpy.mean(dat[:, bu.yi]), Fs = Fs, NFFT = NFFT)
        zpsd, freqs = matplotlib.mlab.psd(dat[:, bu.zi]-numpy.mean(dat[:, bu.zi]), Fs = Fs, NFFT = NFFT)
        xpsd_old, freqs = matplotlib.mlab.psd(dat[:, bu.xi_old]-numpy.mean(dat[:, bu.xi_old]), Fs = Fs, NFFT = NFFT)


	norm = numpy.median(dat[:, bu.zi])
        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,bu.zi])**2
	return [freqs, xpsd, ypsd, dat, zpsd, xpsd_old]

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

        plt.plot(data0[3][:,bu.xi] - np.mean(data0[3][:, bu.xi]) )
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

plot1 = r"1_8E-6mbar_G50_zyNxcool.h5"
plot2 = r"1_8E-6mbar_G50_zyPxcool.h5"
plot3 = r"1_8E-6mbar_G50_zyxcool.h5"
# plot4 = r"2_5E-6mbar_xyzcool_G5.h5"
# plot5 = r"2_4E-6mbar2_xyzcool_G5.h5"

dplot1 = getdata(os.path.join(path,plot1))
dplot2 = getdata(os.path.join(path,plot2))
dplot3 = getdata(os.path.join(path,plot3))
# dplot4 = getdata(os.path.join(path,plot4))
# dplot5 = getdata(os.path.join(path,plot5))

# fig1 = plt.figure()
# plt.subplot(3, 1, 1)
# plt.loglog(dplot1[0][8:],np.sqrt(dplot1[1][8:]), label='Line 2')
# plt.ylabel("V/rtHz")
# plt.subplot(3, 1, 2)
# plt.loglog(dplot2[0][8:],np.sqrt(dplot2[1][8:]))
# plt.subplot(3, 1, 3)
# plt.loglog(dplot3[0][8:],np.sqrt(dplot3[1][8:]))
# plt.ylabel("V/rtHz")
# # plt.subplot(5, 1, 4)
# # plt.loglog(dplot4[0][8:],np.sqrt(dplot4[1][8:]))
# # plt.subplot(5, 1, 5)
# # plt.loglog(dplot5[0][8:],np.sqrt(dplot5[1][8:]))
# plt.xlabel("Frequency[Hz]")
# plt.show()

fig1 = plt.figure()
plt.loglog(dplot1[0][8:],np.sqrt(dplot1[1][8:]), label='New_feedback')
plt.loglog(dplot2[0][8:],np.sqrt(dplot2[1][8:]), label='Old_feedback with Pg')
plt.loglog(dplot3[0][8:],np.sqrt(dplot3[1][8:]), label='Old_feedback')
plt.ylabel("V/rtHz")
plt.xlabel("Frequency[Hz]")
plt.legend(loc='upper left')
plt.ylim(1E-4, 1E-2)
plt.xlim(20, 1E+2)
plt.grid()
plt.show()

# t1 = os.path.getmtime(r"C:\data\20170504\bead4_15um_QWP\new_sensor_feedback\2_8E-6mbar_xyzcool_G5.h5")

# t2 = os.path.getmtime(r"C:\data\20170504\bead4_15um_QWP\new_sensor_feedback\2_7E-6mbar2_xyzcool_G5.h5")

# t3 = os.path.getmtime(r"C:\data\20170504\bead4_15um_QWP\new_sensor_feedback\2_6E-6mbar_xyzcool_G5.h5")

# t4 = os.path.getmtime(r"C:\data\20170504\bead4_15um_QWP\new_sensor_feedback\2_5E-6mbar_xyzcool_G5.h5")

# t5 = os.path.getmtime(r"C:\data\20170504\bead4_15um_QWP\new_sensor_feedback\2_4E-6mbar2_xyzcool_G5.h5")

# print t1 - t2

# print t2 - t3

# print t3 - t4

# print t4 - t5







