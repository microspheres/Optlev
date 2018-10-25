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


path1 = r"C:\data\201712018\bead2_um_QWP_NS_VAT\spin_down_5"


file_list1 = glob.glob(path1+"\*.h5")
file_list1 = list_file_time_order(file_list1)
file_list1 = file_list1

path2 = r"C:\data\201712018\bead2_um_QWP_NS_VAT\spin_down_6"


file_list2 = glob.glob(path2+"\*.h5")
file_list2 = list_file_time_order(file_list2)
file_list2 = file_list2

path3 = r"C:\data\201712018\bead2_um_QWP_NS_VAT\spin_down_7"


file_list3 = glob.glob(path3+"\*.h5")
file_list3 = list_file_time_order(file_list3)
file_list3 = file_list3



Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**16


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

        W = peaks_distance
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
            plt.xlim([peak_pos - 2*peaks_distance, peak_pos + 2*peaks_distance])
            plt.show()

        if(i > 0 or j > 0):
            curr_step = rot-last_peak_pos
            if np.abs(curr_step) < 3*np.abs(last_step):
                peak_pos += (rot - last_peak_pos)
                last_step = rot-last_peak_pos
                last_peak_pos = rot
            else:
                peak_pos += last_step
                last_peak_pos = peak_pos


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
    return [rotation, P, np.array(T)]


c1 = finder(file_list1, 9.99E+5, 8.E4, 1.E4)
c2 = finder(file_list2, 4.17786E+6, 3.E4, 1.E4)
c3 = finder(file_list3, 5.47769E+6, 3.E4, 1.E4)


t0 = c1[2][0]
t = np.hstack([c1[2],c2[2],c3[2]])
rotation = c1[0] + c2[0] + c3[0]
pressures = c1[1] + c2[1] + c3[1]

#  fitting


def func(x, x0, A, tau):
    f = A*(1.0 - 1.0*np.exp(-(x-x0)/tau))
    return f

p0 = [-10., 6.2E7, 500.]
popt, pcov = curve_fit(func, t[120:]-t0, rotation[120:], p0 = np.array(p0))

print popt
print pcov

times =  np.linspace(0, 1460, 6000)


# plt.figure()
# plt.plot(c1[1], c1[0], "bo", label = "QWP = 85.6 deg")

# plt.legend(loc="upper right", frameon = False)
# plt.ylabel("Rotation [Hz]", fontsize=13)
# plt.xlabel("Pressure [mbar]", fontsize=13)
# plt.grid()
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12)
# plt.tight_layout(pad = 0)

plt.figure()
# plt.scatter(np.array(c1[2])-t0, np.array(c1[0])/1000., s=7, c=c1[1], cmap=plt.get_cmap('cool'))
plt.scatter(np.array(t)-t0, np.array(rotation)/1000., s=7, c=pressures, cmap=plt.get_cmap('cool'))
plt.plot(times, func(times, *popt)/1000., "k--")
plt.legend(loc="lower right", frameon = False)
plt.ylabel("Rotation [kHz]", fontsize=13)
plt.xlabel("Time [s]", fontsize=13)
plt.colorbar()
plt.grid()
plt.tight_layout(pad = 0)
plt.show()
