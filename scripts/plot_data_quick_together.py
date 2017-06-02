import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import os, re, time, glob
import bead_util as bu

#refname = r""
#fname0 = r""
#path = r"C:\data\201705010_noise_electric"
path = '/data/20170511/bead2_15um_QWP/new_sensor_feedback/charge43_whole_points/60.0_74.9_75.4'

make_plot_vs_time = True
frequency_list = [41.,43.,47.,49.,53.,57.,67.,71.,73.,79.,83.,89.]
		 

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

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT) 
        drive, freqs = matplotlib.mlab.psd(dat[:, bu.drive]-numpy.mean(dat[:, bu.drive]), Fs = Fs, NFFT = NFFT)
        # zpsd, freqs = matplotlib.mlab.psd(dat[:, bu.zi]-numpy.mean(dat[:, bu.zi]), Fs = Fs, NFFT = NFFT)

	# norm = numpy.median(dat[:, bu.zi])
        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,bu.zi])**2
	# return [freqs, xpsd, ypsd, dat, zpsd]
        return [freqs, xpsd, drive]

#data0 = getdata(os.path.join(path, fname0))

# def rotate(vec1, vec2, theta):
#     vecn1 = numpy.cos(theta)*vec1 + numpy.sin(theta)*vec2
#     vecn2 = numpy.sin(theta)*vec1 + numpy.cos(theta)*vec2
#     return [vec1, vec2]



def list_file_time_order(p):
    filelist = glob.glob(os.path.join(p,"*.h5"))
    filelist.sort(key=os.path.getmtime)
    return filelist



#data = map(getdata,list_file_time_order(path))

F = np.zeros(NFFT/2 + 1)
X = np.zeros(NFFT/2 + 1)
driveX = np.zeros(NFFT/2 + 1)

for file in list_file_time_order(path)[:]:
    a = getdata(file)
    F = a[0]
    X += a[1]
    driveX += a[2]


plt.loglog(F,X/len( list_file_time_order(path)[:]))
plt.loglog(F,driveX/len( list_file_time_order(path)[:]))
yvalues = plt.ylim()
clist = bu.get_color_map( len(frequency_list) )
# for i,f in enumerate(frequency_list):
#         plt.plot([f,f],yvalues,color = clist[i])
#         plt.plot([2*f,2*f],yvalues,color = clist[i])
plt.grid()
plt.show()





