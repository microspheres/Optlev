import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt

folder = r"C:\data\20191106\test\fpga_transfer_funct"

filelist = glob.glob(folder+"\*.h5")

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
                PID = dset.attrs['PID']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

        chirp = dat[:, 0]-numpy.mean(dat[:, 0])
        response = dat[:, 7]-numpy.mean(dat[:, 7])

	return [chirp, response, Fs]

def transfer(filename):

    chirp, response, Fs = getdata(filename)

    Fs = int(Fs)
    
    fft_c = np.fft.rfft(chirp, chirp.size)
    fft_r = np.fft.rfft(response, response.size)
    freq = np.fft.rfftfreq(chirp.size, 1./Fs)

    t = np.abs(fft_r/fft_c)

    arg_c = 1.*np.angle(fft_c)
    arg_r = 1.*np.angle(fft_r)

    
    angle = arg_r - arg_c
    angle = ( angle + np.pi) % (2 * np.pi ) - np.pi


    return [freq, t, angle]

for i in filelist:
    
    f, t, a = transfer(i)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title(str(i[42:-3]))
    plt.loglog(f, t)
    plt.xlim(10, 1500)
    plt.ylim(1e-2, 5)
    plt.ylabel("|transfer funtion|")
    plt.xlabel("freq [Hz]")
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.semilogx(f, a)
    plt.ylabel("Angle [rad]")
    plt.xlabel("freq [Hz]")
    plt.ylim(-np.pi, np.pi)
    plt.xlim(10, 3500)
    plt.grid()
    name1 = "transfer_function_" + str(i[42:-3]) + ".pdf"
    name1 = os.path.join(folder, name1)
    plt.savefig(name1)



plt.show()



