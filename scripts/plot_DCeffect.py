import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import re

path = r"C:\data\20170925\bead4_15um_QWP_NS\DC"
file_list = glob.glob(path+"\*.h5")

make_plot_vs_time = True
conv_fac = 4.4e-14
		

final_index = -1
		 

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
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:final_index, bu.xi]-numpy.mean(dat[:final_index, bu.xi]), Fs = Fs, NFFT = NFFT) 
	ypsd, freqs = matplotlib.mlab.psd(dat[:final_index, bu.yi]-numpy.mean(dat[:final_index, bu.yi]), Fs = Fs, NFFT = NFFT)
        zpsd, freqs = matplotlib.mlab.psd(dat[:final_index, bu.zi]-numpy.mean(dat[:final_index, bu.zi]), Fs = Fs, NFFT = NFFT)
        xpsd_old, freqs = matplotlib.mlab.psd(dat[:final_index, bu.xi_old]-numpy.mean(dat[:final_index, bu.xi_old]), Fs = Fs, NFFT = NFFT)
        # Ddrive = dat[:, bu.drive]*np.gradient(dat[:,bu.drive])
        # DdrivePSD, freqs =  matplotlib.mlab.psd(Ddrive-numpy.mean(Ddrive), Fs = Fs, NFFT = NFFT)

        x = dat[:final_index,bu.xi] - np.mean(dat[:final_index,bu.xi])
        v = np.mean(dat[:,bu.drive])
	norm = numpy.median(dat[:, bu.zi])
        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,bu.zi])**2
        return [freqs, xpsd, ypsd, dat, zpsd, x, xpsd_old,v]

def get_voltage(name):
    i = re.findall("-?\d+mVdc",name)[0][:-4]
    return int(i)

def get_index(name):
    # i = re.findall("\d+.h5",name)[0][:-3]
    i = 0
    return int(i)


def DC_sort(file_list):
    return_list = []
    dc_val = []
    for i in file_list:
        dc_val.append(get_voltage(i))
    dc_val = np.unique(dc_val)
    for dc in dc_val:
        # if dc not in [2,13]: continue:
        aux = []
        for i in file_list:
            if get_voltage(i) == dc:
                aux.append(i)
        aux = sorted(aux,key = get_index)
        return_list.append(aux)
    return return_list

plt.figure()
# diff = []
psds = []

X = []

for i in DC_sort(file_list):
    x = 0
    f = getdata(i[0])[0]
    v = getdata(i[0])[-1]
    for j in i:
        x += getdata(j)[1]/len(i)
    plt.loglog(f, x, label = str('%.1f' % round(200.0*v,1))+" DC Volt")
    # diff.append(x)
    psds.append(x)

plt.ylabel('PSD [V^2/Hz]')
plt.xlabel('Freq [Hz]')
plt.legend()
# plt.figure()
# ndiff = np.abs(diff[0]-diff[1])
# plt.loglog(f,ndiff)





# plt.figure()
# for i in DC_sort(file_list):
#     xx = 0
#     v = getdata(i[0])[-1]
#     for j in i:
#         xx = getdata(j)[5]
#     plt.plot(xx, label = str('%.1f' % round(200.0*v,1))+" DC Volt")
#     # diff.append(x)
#     X.append(xx)

# plt.ylabel('X_signal [V]')
# plt.xlabel('time')
# plt.legend()
# # plt.figure()
# ndiff = np.abs(diff[0]-diff[1])
# plt.loglog(f,ndiff)
plt.show()
