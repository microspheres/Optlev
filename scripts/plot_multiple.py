import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob


def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist


path = r"C:\data\20180109\bead15_um_POL_NS_VAT\pressures"
		

file_list = glob.glob(path+"\*.h5")

file_list = list_file_time_order(file_list)

file_list = file_list[-20:]
		 

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
                Press = dset.attrs['temps'][0]
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2] )

        xpsd_old, freqs = matplotlib.mlab.psd(dat[:, 0]-numpy.mean(dat[:, 0]), Fs = Fs, NFFT = NFFT)
        # Ddrive = dat[:, bu.drive]*np.gradient(dat[:,bu.drive])
        # DdrivePSD, freqs =  matplotlib.mlab.psd(Ddrive-numpy.mean(Ddrive), Fs = Fs, NFFT = NFFT)

        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,bu.zi])**2
	return [freqs, 0, 0, 0, 0, xpsd_old, Press]


freq = getdata(file_list[0])[0]
cmap = bu.get_color_map(len(file_list))
for i,c in zip(file_list, cmap):
        aux_press = str("mbar")
        aux = str(getdata(i)[6])
        aux_angle = str("_angle=")
        aux2 = i[i.rfind('_')+1:i.rfind('.h5')]
        plt.loglog(freq, getdata(i)[5], label = aux+aux_press+aux_angle+aux2, color = c)
        plt.legend(loc="upper right", frameon = False)
plt.ylabel(r'Sensor response [$V^2/Hz$]', fontsize = 20)
plt.xlabel(r'Frequency [Hz]', fontsize = 20)
plt.rcParams.update({'font.size':20})
plt.show()
