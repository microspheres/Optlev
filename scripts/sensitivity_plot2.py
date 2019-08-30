import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os, re
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

several_AC = False
several_DC = False
do_fit = True
save_figure = True

path_charge = r"C:\data\20190812\22um_SiO2\4\calibration1e"
file_list_charge = glob.glob(path_charge+"\*.h5")

path_psd = r"C:\data\20190812\22um_SiO2\4\calibration1e"
file_list_psd = glob.glob(path_psd+"\*.h5")

path_save = path_psd

rho = 1800.0

R = 11.*10**-6

mass = (4./3.)*np.pi*(R**3)*rho

Number_of_e = (7.76*10**14)

v_calibration = 0.9 # vpp in the daq
v_calibration = v_calibration/2.0 # now v is in amplitude

distance = 0.01 #m

VAC = 20. # V. only used for few applications. 

NFFT = 2**17

startfile = 0
endfile = -1

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

file_list_psd = list_file_time_order(file_list_psd)

file_list_psd = file_list_psd[:]

drive_daq = 3
p = bu.xi
label_unit = 'X_signal_units_2.pdf'
label_save = "f_and_2f_arbunits_X_axis.txt"
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
        drivepsd, freqs = matplotlib.mlab.psd(dat[:, drive_daq]-numpy.mean(dat[:, drive_daq]), Fs = Fs, NFFT = NFFT)
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
    ratio_voltage = v_calibration/VAC # is the best the trek can do (in amplitude)
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
    arg = getdata(file_list_charge[0])[4]
    arg2 = 2*arg
    c = v_to_electron(file_list_charge)
    plt.figure()
    plt.ylabel('electron number/$\sqrt{Hz}$')
    plt.xlabel('Freq[Hz]')
    plt.loglog(A[0], c*np.sqrt(A[1])) # the 0.5 factor comes from the consideration that the neutron counts as 1 pronton and 1 electron. not currently used
    np.savetxt(os.path.join(path_save,'freq_vs_electron_number.txt'), (A[0],c*np.sqrt(A[1])))
    plt.loglog(A[0], (np.sqrt(A[0][1]-A[0][0]))*c*np.sqrt(A[1]))
    plt.plot(A[0][arg], (np.sqrt(A[0][1]-A[0][0]))*c*np.sqrt(A[1][arg]), "ro")
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

def plot_in_signal_units(file_list_psd, file_list_charge): # check that the sensor gain for calibration is the same for the measurement!
    # A = unite_psd(file_list_psd)
    c = 1.0
    
    arg = getdata(file_list_charge[0])[4]
    freqpeak = A[0][arg]
    freqpeak2 = A[0][2*arg]
    
    plt.figure()
    plt.ylabel('Signal Units/$\sqrt{Hz}$')
    plt.xlabel('Freq[Hz]')
    plt.loglog(A[0], c*np.sqrt(A[1]))
    # plt.axvline(freqpeak, color='k', linestyle='--', lw = 0.5)
    # plt.plot(freqpeak, c*np.sqrt(A[1][arg]), "ro")
    # # plt.axvline(freqpeak2, color='k', linestyle='--', lw = 0.5)
    # plt.plot(freqpeak2, c*np.sqrt(A[1][2*arg]), "ro")
    # print "value of the 2f peak in arb units", c*np.sqrt(A[1][2*arg])
    # print "value of the 2f peak in arb units", c*np.sqrt(A[1][2*arg-1])
    plt.xlim(10, 1000)
    plt.grid()
    plt.savefig(os.path.join(path_save, label_unit))
    print "f", c*np.sqrt(A[1][arg])
    print "2f", c*np.sqrt(A[1][2*arg-1])
    # np.savetxt(os.path.join(path_save,'freq_vs_arb.txt'), (A[0],c*np.sqrt(A[1])))
    np.savetxt(os.path.join(path_save, label_save), (A[0],c*np.sqrt(A[1])))
    return


###################################### several AC
def several_AC_psdlist(file_list_psd, AC_list):
    if not several_AC and not several_DC:
        return "several_AC = False"
    L = [[] for i in range(len(AC_list))] # list of list

    for v in range(len(AC_list)):
        for i in range(len(file_list_psd)):
            if( several_AC ):
                a = float(file_list_psd[i].split("synth")[1].split("mV")[0]) #daq AC voltage in mV
            elif (several_DC):
                a = float(re.findall("-?\d+mVdc",file_list_psd[i])[0][:-4])
            if AC_list[v] == a:
                b = file_list_psd[i]
                L[v].append(b)
    return L

def get_peaks_AC_list(file_list_psd, file_list_charge, AC_list): # it outputs the signal/Hz for the peak and 2x peak
    if not several_AC and not several_DC:
        return "several_AC = False"


    arg = getdata(file_list_charge[0])[4]
    arg2 = 2*arg
    argnoise = 2*arg-10
    
    L = several_AC_psdlist(file_list_psd, AC_list)
    
    A = [[] for i in range(len(AC_list))] # list of list
    A2 = [[] for i in range(len(AC_list))] # list of list
    AN = [[] for i in range(len(AC_list))] # list of list
    
    for i in range(len(AC_list)):
        a = unite_psd(L[i])[1]
        ax = a[arg]
        ax2 = a[arg2]
        an = np.mean(a[argnoise-3:argnoise+3])
        A[i].append(ax)
        A2[i].append(ax2)
        AN[i].append(an)

    return [A,A2,AN]
    
def plot_peaks_several_AC(file_list_psd, file_list_charge, AC_list):
    from scipy.optimize import curve_fit
    if not several_AC and not several_DC:
        return "several_AC = False"
    
    A, A2, AN = get_peaks_AC_list(file_list_psd, file_list_charge, AC_list)
    A = np.ndarray.flatten(np.array(A))
    A2 = np.ndarray.flatten(np.array(A2))

    def line(x,a,b):
        return a*x + b

    def para(x,a,b):
        return a*(x**2) + b    
    
    p1 = np.array([1.5e-6, 5.0e-4])
    if(do_fit):
        popt1, pcov1 = curve_fit(line, (200/1000.)*np.array(AC_list), np.sqrt(np.array(A)), p0 = p1)
        popt2, pcov2 = curve_fit(para, (200/1000.)*np.array(AC_list), np.sqrt(np.array(A2)))
    else:
        popt1, pcov1 = p1, np.zeros((len(p1),len(p1)))
        popt2, pcov2 = p1, np.zeros((len(p1),len(p1)))
    
    volt = np.linspace(0, max((200/1000.)*np.array(AC_list)) + 100, 10)

    plt.figure()
    plt.plot((200/1000.)*np.array(AC_list), np.sqrt(np.array(A)), "ro", label = "$\omega$")
    plt.plot((200/1000.)*np.array(AC_list), np.sqrt(np.array(A2)), "ko", label = "2$\omega$")
    plt.plot((200/1000.)*np.array(AC_list), np.sqrt(np.array(AN)), "g*", label = "2$\omega$ - 10freqbin_avg")

    if(do_fit):
        plt.plot(volt, line(volt, *popt1), "r--")
        plt.plot(volt, para(volt, *popt2), "k--")

    plt.ylabel("Signal units/$\sqrt{Hz}$")
    if( several_AC ):
        plt.xlabel("AC Voltage pp [V]")
    else:
        plt.xlabel("DC Voltage offset [V]")

    plt.legend()
    plt.grid()
    plt.tight_layout(pad = 0)
    if(save_figure):
        plt.savefig(os.path.join(path_save,'peaks.pdf'))
    return



    

AC_list = [1000., 2750., 4500., 6250., 8000.]

#plot_peaks_several_AC(file_list_psd, file_list_charge, AC_list)


A = unite_psd(file_list_psd)


# plot_sensitivity_force(file_list_psd, file_list_charge)
plot_sensitivity_electron(file_list_psd, file_list_charge)
plot_in_signal_units(file_list_psd, file_list_charge)
plot_sensitivity_g(file_list_psd, file_list_charge)
# plot_sensitivity_acc(file_list_psd, file_list_charge)

# plt.figure()
# plt.plot(A[2])


# path_psd = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\electrode_dist_fartherN4\meas2_no_field"
# file_list_psd = glob.glob(path_psd+"\*.h5")

# A = unite_psd(file_list_psd)
# plot_sensitivity_electron(file_list_psd, file_list_charge)

plt.show()
