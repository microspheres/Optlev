import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt
from scipy.signal import butter, lfilter, filtfilt
def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

#folder_meas = r"C:\data\20191122\10um\2\charge_plot"
folder_meas1 = r"C:\data\20200114\10um\4\charge2"
folder_meas2 = r"C:\data\20200114\10um\4\charge3"
folder_meas3 = r"C:\data\20200114\10um\4\charge4"
folder_meas_HF = r"C:\data\20200114\10um\4\charge1"

show_aux_plots = False

distance = 0.0033

drive_col = 3

NFFT = 2**13

F_res = 52. # the sphere resonance freq
#synth_HF = "50ohm" # choose between "50ohm" and "High Z"

def getdata(fname):
	# print "Opening file: ", fname
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
                time = dset.attrs['Time']

                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xin = dat[:, 0]-numpy.mean(dat[:, 0])
        xout = dat[:, 4]-numpy.mean(dat[:, 4])
        drive = dat[:, drive_col]-numpy.mean(dat[:, drive_col])

	return [xin, xout, drive, Fs, time]

def plot_charge(folder_meas, t0, f_res): # return freq, freq_arg, amplitude of the drive and sphere x motion in the freq band. The assumption is the synth with high Z is used.
    file_list = glob.glob(folder_meas+"\*.h5")
    file_list = list_file_time_order(file_list)

    i = file_list[10]
    a = getdata(i)

    drive = a[2]
    Fs = a[3]
    xin = a[0]
    xout = a[1]

    freq, drive_psd = sp.csd(drive, drive, Fs, nperseg=NFFT, scaling = "spectrum")

    
    f0arg = np.argmax(drive_psd)
    f0 = freq[f0arg]

    if f0 > f_res:
        a = "abort, def plot_charge can only accept freq of the field that are smaller than the sphere res freq"
        print a
        return 

    if show_aux_plots:
        plt.figure()
        plt.loglog(freq, drive_psd)
        plt.loglog(freq[f0arg], drive_psd[f0arg], "ro")
#    plt.show()

    Charge = []
    Time = []
    V = []
    F = []
    
    for j in file_list:
        a = getdata(j)
        time = a[4]
        
        drive = a[2]
        Fs = a[3]
        xin = a[0]
        xout = a[1]

        # freq, test = sp.csd(xin, xin, Fs, nperseg=NFFT)
        # test = np.sqrt(test)
        # plt.loglog(freq, test)
        # plt.loglog(freq[f0arg], test[f0arg], "ro") 
        
        freq, charge = sp.csd(xin, drive, Fs, nperseg=NFFT, scaling = "spectrum")
        freq, v2 = sp.csd(drive, drive, Fs, nperseg=NFFT, scaling = "spectrum")
        n = np.sign(np.real(charge[f0arg]))
        charge = n*np.abs(charge[f0arg])/( np.abs( np.real( v2[f0arg] ) )) # abs(charge[f0arg]) is because at high freq, the phase gets scrambed due to the feedback. and n is for the sign
        Charge.append(charge)
        Time.append(time)
        v = (2.*np.abs(v2[f0arg]))**0.5
        V.append(v)
        F.append(freq[f0arg])

    Time = np.array(Time)
    Time = Time - Time[0]

    index = np.where(Time >= t0)[0][0]
    
    Charge = np.array(Charge)


    V = np.array(V)
    print "force 1e = ", 200.*1.6e-19*v/distance

    p1 = np.abs(np.mean(Charge[0:5])/3.) # this depends on the specific folder
    Charge = Charge[index:]/p1
    Time = Time[index:] - Time[index:][0]
    V = V[index:]

    if show_aux_plots:
           plt.figure(figsize=(5.5,3))
           plt.rcParams.update({'font.size': 14})
           plt.plot(Time, Charge, "r.")

           plt.plot(Time, V, "b.")
           plt.xlabel("Time [S]")
           plt.ylabel("Charge [e$^-$]")
           # plt.title(title)
           plt.grid(which='both')
           plt.tight_layout(pad = 0)
    return [Charge, Time, V, p1]


def plot_charge_after(folder_meas, t0, tf, f_res, conversion, synth_HF): # the plot_charge defines the conversion for this code and must run before!
    file_list = glob.glob(folder_meas+"\*.h5")
    file_list = list_file_time_order(file_list)

    try:
        i = file_list[80]
    except:
        i = file_list[10]
    a = getdata(i)
    drive = a[2]
    Fs = a[3]
    xin = a[0]
    xout = a[1]

    freq, drive_psd = sp.csd(drive, drive, Fs, nperseg=NFFT, scaling = "spectrum")    
    f0arg = np.argmax(drive_psd)
    f0 = freq[f0arg]

    if show_aux_plots:
        plt.figure()
        plt.loglog(freq, drive_psd)
        plt.loglog(freq[f0arg], drive_psd[f0arg], "ro")
#    plt.show()

    Charge = []
    Time = []
    V = []
    F = []
    for j in file_list:
        a = getdata(j)
        time = a[4]

        if synth_HF == "50ohm":
            drive = a[2]*0.5 # the 0.5 factor is due to the impedance used.
        if synth_HF == "High Z":
            drive = a[2]
        Fs = a[3]
        xin = a[0]
        xout = a[1]

        # freq, test = sp.csd(xin, xin, Fs, nperseg=NFFT)
        # test = np.sqrt(test)
        # plt.loglog(freq, test)
        # plt.loglog(freq[f0arg], test[f0arg], "ro") 
        
        freq, charge = sp.csd(xin, drive, Fs, nperseg=NFFT, scaling = "spectrum")
        freq, v2 = sp.csd(drive, drive, Fs, nperseg=NFFT, scaling = "spectrum")
        n = np.sign(np.real(charge[f0arg]))
        charge = n*np.abs(charge[f0arg])/( np.abs( np.real( v2[f0arg] ) )) # abs(charge[f0arg]) is because at high freq, the phase gets scrambed due to the feedback. and n is for the sign
        Charge.append(charge)
        Time.append(time)
        v = (2.*np.abs(v2[f0arg]))**0.5
        V.append(v)
        F.append(freq[f0arg])

    Time = np.array(Time)
    Time = Time - Time[0]

    index = np.where(Time >= t0)[0][0]
    try:
        index2 = np.where(Time >= tf)[0][0]
    except:
        index2 = len(Time)-1
    
    Charge = np.array(Charge)
    V = np.array(V)
    
    if f0 > f_res:
        Charge = (Charge[index:index2]/conversion)*((freq[f0arg]/f_res)**2)
    if f0 < f_res:
        Charge = (Charge[index:index2]/conversion)

    Time = Time[index:index2] - Time[index:index2][0]
    V = V[index:index2]
    if show_aux_plots:
        plt.figure(figsize=(5.5,3))
        plt.rcParams.update({'font.size': 14})
        plt.plot(Time, Charge, "r.")
        
        plt.plot(Time, V, "b.")
        plt.xlabel("Time [S]")
        plt.ylabel("Charge [e$^-$]")
        # plt.title(title)
        plt.grid(which='both')
        plt.tight_layout(pad = 0)
    return [Charge, Time, V]



step = plot_charge(folder_meas3, 0, F_res)
HF = plot_charge_after(folder_meas_HF, 50, 1430+50, F_res, step[3], "50ohm")
a = plot_charge_after(folder_meas1, 0, 1430+50, F_res, step[3], "High Z")
b = plot_charge_after(folder_meas2, 0, 1430+50, F_res, step[3], "High Z")

# time order = HF, a, b, step

a[1] = HF[1][-1] + a[1] 
b[1] = a[1][-1] + b[1]
step[1] = b[1][-1] + step[1]

HF[0] = list(HF[0])
a[0] = list(a[0])
b[0] = list(b[0])
step[0] = list(step[0])

HF[1] = list(HF[1])
a[1] = list(a[1])
b[1] = list(b[1])
step[1] = list(step[1])

ch = HF[0] + a[0] + b[0] + step[0]
t = HF[1] + a[1] + b[1] + step[1]

ch1 = HF[0][664:] + a[0] + b[0] + step[0]
t1 = HF[1][664:] + a[1] + b[1] + step[1]

ch2 = step[0]
t2 = step[1]

import matplotlib.ticker as mticker

fig = plt.figure()
plt.rcParams.update({'font.size': 10})
ax = fig.add_subplot(131)
ax.plot(np.array(t)/60., ch, "r.")
ax.set_yticks([10e3, 5e3, 0])
ax.set_yticklabels(["10000", "5000", "0"])
ax.set_ylabel("Charge [units of e$^-$]")
ax.grid()

ax2 = fig.add_subplot(132)
ax2.set_yticks([50, 0, -50])
ax2.plot(np.array(t1)/60., ch1, "r.")
ax2.set_xlabel("Time [min]")
ax2.grid()

ax3 = fig.add_subplot(133)
ax3.plot(np.array(t2)/60., ch2, "r.")
ax3.set_yticks([0, -1, -2, -3])
fig.set_size_inches(6,2.1)
ax3.grid()

plt.tight_layout(pad = 0)

plt.show()
