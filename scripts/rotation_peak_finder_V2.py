import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import bead_util as bu
import glob
from scipy.optimize import curve_fit


def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist


path1 = r"C:\data\20180129\bead1_um_POL_NS_SiO2_10um\meas_1"


file_list1 = glob.glob(path1+"\*.h5")
file_list1 = list_file_time_order(file_list1)
file_list1 = file_list1


# Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**10


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
                time = dset.attrs['Time']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2] )

        xpsd_old, freqs = matplotlib.mlab.psd(dat[:, 0]-numpy.mean(dat[:, 0]), Fs = Fs, NFFT = NFFT)
        # Ddrive = dat[:, bu.drive]*np.gradient(dat[:,bu.drive])
        # DdrivePSD, freqs =  matplotlib.mlab.psd(Ddrive-numpy.mean(Ddrive), Fs = Fs, NFFT = NFFT)

        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,bu.zi])**2
	return [freqs, 0, 0, 0, 0, xpsd_old, Press, time]


def return_arg(list, value):
    aux = 1.0*np.array(list) - 1.0*value
    aux = np.abs(aux)
    arg = np.argmin(aux)
    return arg

N = 1
def finder(filelist, peak_pos, peaks_distance, peak_step):
    freq = getdata(filelist[0])[0]

    rotation = []
    P = []
    T = []

    last_peak_pos = peak_pos
    last_step = peak_step

    for i in range(len(filelist)/N):
        
        data = np.zeros(len(freq))

        for j in range(N):
            gd = getdata(filelist[i*N + j])
            data += gd[5]
        data /= N

        W = peaks_distance*peak_pos
        argpeak = return_arg(freq, peak_pos)
        argW = return_arg(freq, W)
        
        Peakrange = data[(argpeak - argW):(argpeak + argW)]

        shortfreq = freq[(argpeak - argW):(argpeak + argW)]
        lin = np.polyfit(shortfreq, Peakrange, 1)

        subtract_data = Peakrange - np.polyval(lin, shortfreq)
        Peak = np.argmax(subtract_data)
        
        if False:
            plt.figure()
            plt.plot(shortfreq,subtract_data)
            plt.plot(shortfreq, 5*np.ones(len(shortfreq))*np.std(subtract_data))
            plt.show()
        
        #if (len( np.argwhere(subtract_data > 7.*np.std(subtract_data))) > 1):
        #if ( i>0 and  np.abs(shortfreq[Peak] - peak_pos) > 50*np.abs(peak_pos - last_peak_pos) ):
        #    rot = -1
        #else:
        rot = shortfreq[Peak]
        

        if False:
            plt.figure()
            plt.loglog(freq,data)
            plt.plot(shortfreq, data[(argpeak - argW):(argpeak + argW)])
            plt.plot(shortfreq[Peak], data[(argpeak - argW):(argpeak + argW)][Peak], "rx")
            #plt.xlim([peak_pos - 2*peaks_distance, peak_pos + 2*peaks_distance])
            plt.show()

        if(i > 0 ):
            curr_step = rot-last_peak_pos
        else:
            curr_step = 10*last_step
        
        if np.abs(curr_step) < 5*np.abs(last_step):
            peak_pos = rot
            last_step = curr_step
            last_peak_pos = rot
        else:
            peak_pos = rot #+= last_step
            last_peak_pos = peak_pos

        # print peak_pos, rot, last_peak_pos

        #print peak_pos, last_peak_pos, rot     
        #if rot > 0:
        #    if i>0:
        #        last_peak_pos = rot
        #else:
        #    temp_pos = last_peak_pos
        #    peak_pos = 2.*peak_pos - temp_pos
        #    last_peak_pos = peak_pos
        #print peak_pos, last_peak_pos, rot
        #raw_input()
            
        rotation.append(rot)
        P.append(gd[6])
        T.append(gd[7])
    return [rotation, P, np.array(T)-T[0]]



c1 = finder(file_list1, 820000., 0.8, 20000./4)


#  fitting


def func(x, x0, A, tau):
    f = A*(1.0 - np.exp(-(x-x0)/tau))
    return f

def func2(X, x0, A, B): # takes pressure into account
    x, p = X
    f = A*p*(1.0 - np.exp(-(x-x0)*B/p))
    return f

def funcdecay(x, x0, A, tau, v0):
    f = A + v0*np.exp(-(x-x0)/tau)
    return f

def funcdecay2(X, x0, A, B): # takes pressure into account
    x, p = X
    f = A*p*(np.exp(-(x-x0)*B/p))
    return f

# p0 = [-510.0 ,6.27E6, 11000]
# popt, pcov = curve_fit(func, c1[2][150:200], c1[0][150:200], p0 = np.array(p0))


# print popt
# print pcov


times =  np.linspace(11000, 21000, 10000)
times2 =  np.linspace(20500, 28000, 10000)
times3 =  np.linspace(28000, 32000, 10000)
pressures = np.linspace(1E-5, 1E-7, 10000)


plt.figure()
plt.scatter(c1[2], c1[0], s=3, c=np.log10(c1[1]), label = "", vmin = -6.5, vmax = -6.25)
plt.plot(times, func(times, *popt), "r--", label = "")
# plt.plot(times, func2((times, pressures), *popt2), "g--", label = "")

plt.plot(times2, funcdecay(times2, *popt3), "r--", label = "")
# plt.plot(times3, funcdecay(times3, *popt4), "r--", label = "")

plt.grid()

plt.ylabel("Rotation [Hz]", fontsize=13)
plt.xlabel("Time [s]", fontsize=13)

plt.colorbar()
plt.show()

# plt.figure()
# plt.plot(c1[2], c1[0], "b.")

# plt.legend(loc="upper right", frameon = False)

# plt.grid()
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12)
# plt.tight_layout(pad = 0)
