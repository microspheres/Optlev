import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob


path_charge = r"C:\data\20171110\bead3_15um_QWP_NS\calibration\1e"
file_list_charge = glob.glob(path_charge+"\*.h5")

path_psd = r"C:\data\20171110\bead3_15um_QWP_NS\meas\AC_only4"
file_list_psd = glob.glob(path_psd+"\*.h5")

path_save = r"C:\data\20171110\bead3_15um_QWP_NS\meas\AC_only4"

mass = (1.19*10**-12) # in kg

Number_of_e = (3.58*10**14)

v_calibration = 0.1 # vpp in the daq
v_calibration = v_calibration/2.0 # now v is in amplitude

distance = 0.0021 #m

NFFT = 2**19

startfile = 0
endfile = -1

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

file_list_psd = list_file_time_order(file_list_psd)

file_list_psd = file_list_psd[startfile:endfile]


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
        drivepsd, freqs = matplotlib.mlab.psd(dat[:, bu.drive]-numpy.mean(dat[:, bu.drive]), Fs = Fs, NFFT = NFFT)
        aux = np.argmax(drivepsd)
        freq_drive = freqs[aux]
	return [freqs, xpsd, drivepsd, freq_drive, aux]

def unite_psd(file_list): # for charge calibration
    freqs = np.array(getdata(file_list[0])[0])
    X = np.zeros(len(freqs))
    noise_vs_time = []
    xindex = np.argmin(np.abs(freqs-48.))
    for file in file_list:
       a = getdata(file)
       X += np.array(a[1])
       noise_vs_time.append(a[1][xindex])
    return [freqs, X/len(file_list),noise_vs_time]

def Voltsquare_at_peak(file_list): # area of the peak
    a = 2
    freq, xpsd, b = unite_psd(file_list)
    peak = getdata(file_list[0])[4]
    dfreq = freq[peak] - freq[peak - 1]
    v2 = np.sum(xpsd[peak - a:peak + a])*dfreq
    return v2

def v_to_newton(file_list):
    v = 200.0*v_calibration # volts
    E = 1.0*v/distance
    charge = 1.602*10**(-19) # SI units
    force = charge*E
    conversion = force/np.sqrt(Voltsquare_at_peak(file_list))
    return conversion

def v_to_g(file_list):
    v = 200.0*v_calibration # volts
    E = 1.0*v/distance
    charge = 1.602*10**(-19) # SI units
    force = charge*E
    a = force/mass
    conversion = (a/np.sqrt(Voltsquare_at_peak(file_list)))/9.8
    return conversion

def v_to_acc(file_list):
    v = 200.0*v_calibration # volts
    E = 1.0*v/distance
    charge = 1.602*10**(-19) # SI units
    force = charge*E
    a = force/mass
    conversion = (a/np.sqrt(Voltsquare_at_peak(file_list)))
    return conversion

def v_to_electron(file_list):
    conversion = 1/np.sqrt(Voltsquare_at_peak(file_list))
    N = Number_of_e
    ratio_voltage = v_calibration/10.0 # is the best the trek can do (in amplitude)
    return (conversion/N)*ratio_voltage

def plot_sensitivity_force(file_list_psd, file_list_charge): # check that the sensor gain for calibration is the same for the measurement!
    # A = unite_psd(file_list_psd)
    c = v_to_newton(file_list_charge)
    plt.figure()
    plt.ylabel('Force[N]/$\sqrt{Hz}$')
    plt.xlabel('Freq[Hz]')
    plt.loglog(A[0], c*np.sqrt(A[1]))
    plt.grid()
    np.savetxt(os.path.join(path_save,'freq_vs_force[N].txt'), (A[0],c*np.sqrt(A[1])))
    return

def plot_sensitivity_electron(file_list_psd, file_list_charge): # check that the sensor gain for calibration is the same for the measurement!
    # A = unite_psd(file_list_psd)
    c = v_to_electron(file_list_charge)
    plt.figure()
    plt.ylabel('electron number/$\sqrt{Hz}$')
    plt.xlabel('Freq[Hz]')
    plt.loglog(A[0], c*np.sqrt(A[1])) # the 0.5 factor comes from the consideration that the neutron counts as 1 pronton and 1 electron. not currently used
    np.savetxt(os.path.join(path_save,'freq_vs_electron_number.txt'), (A[0],c*np.sqrt(A[1])))
    plt.grid()
    return

def plot_sensitivity_g(file_list_psd, file_list_charge): # check that the sensor gain for calibration is the same for the measurement!
    # A = unite_psd(file_list_psd)
    c = v_to_g(file_list_charge)
    plt.figure()
    plt.ylabel('acceleration [g]/$\sqrt{Hz}$')
    plt.xlabel('Freq[Hz]')
    plt.loglog(A[0], c*np.sqrt(A[1]))
    plt.grid()
    np.savetxt(os.path.join(path_save,'freq_vs_g.txt'), (A[0],c*np.sqrt(A[1])))
    return

def plot_sensitivity_acc(file_list_psd, file_list_charge): # check that the sensor gain for calibration is the same for the measurement!
    # A = unite_psd(file_list_psd)
    c = v_to_acc(file_list_charge)
    plt.figure()
    plt.ylabel('acceleration [m/$s^2$]/$\sqrt{Hz}$')
    plt.xlabel('Freq[Hz]')
    plt.loglog(A[0], c*np.sqrt(A[1]))
    # plt.ylim(5e-6, 4e-2)
    # plt.xlim(0, 3000)
    plt.grid()
    np.savetxt(os.path.join(path_save,'freq_vs_acc[SI].txt'), (A[0],c*np.sqrt(A[1])))
    return

A = unite_psd(file_list_psd)


plot_sensitivity_force(file_list_psd, file_list_charge)
plot_sensitivity_electron(file_list_psd, file_list_charge)
plot_sensitivity_g(file_list_psd, file_list_charge)
plot_sensitivity_acc(file_list_psd, file_list_charge)

plt.figure()
plt.plot(A[2])

plt.show()
