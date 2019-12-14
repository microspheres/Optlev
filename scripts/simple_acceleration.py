import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt

folder_calibration = r"C:\data\201908020\22um_SiO2_pinhole\5\calibration1p"
folder_calibration = r"C:\data\20190912\prechamber\2\new_FB\calibration1e"

folder_meas = folder_calibration
# folder_meas = r"C:\data\20190912\prechamber\2\new_FB\meas_no_field"

folder_hp = r"C:\data\201908020\22um_SiO2_pinhole\5"
folder_hp = r"C:\data\20190912\prechamber\2\HP"
file_high_pressure = r"5mbar_zcool_0.h5"

file_list = glob.glob(folder_calibration+"\*.h5")

NFFT = 2**16

drive_col = 3

Diameter = 22.8e-6 # meters

rho = 1800

d = 0.0097
#d = 0.005

number_of_charge = 1.0

elec_charge = number_of_charge*(1.60218e-19)

def mass(Diameter, rho):
    m = (4/3.)*(np.pi)*((Diameter/2)**3)*rho
    return m

mass = mass(Diameter, rho)
print mass

freq_plot_min = 10.
freq_plot_max = 300.

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
                pid = dset.attrs['PID']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT)
        drivepsd, freqs = matplotlib.mlab.psd(dat[:, drive_col]-numpy.mean(dat[:, drive_col]), Fs = Fs, NFFT = NFFT)


	return [freqs, xpsd, drivepsd]

def psd(f, A, f0, gamma):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    gamma1 = 2.0*np.pi*gamma
    s1 = 2.*A*(gamma1*(w0**2))
    s2 = 1.*(w0**2)*((w0**2 - w**2)**2 + (gamma1*w)**2)
    s = np.sqrt(s1/s2)
    return A*s

def find_resonance(folder_hp, file_high_pressure):
    a = getdata(os.path.join(folder_hp, file_high_pressure))
    freq = a[0]
    xpsd = np.sqrt(a[1])
    popt, pcov = opt.curve_fit(psd, freq[60:1000], xpsd[60:1000], p0 = [0.01, 85, 10])
    # plt.figure()
    # plt.loglog(freq[60:1000], xpsd[60:1000])
    # plt.loglog(freq, psd(freq, *popt))
    # plt.show()
    return [popt[1], freq, xpsd]


def get_drive_and_motion(folder_calibration): # return freq, freq_arg, amplitude of the drive and sphere x motion in the freq band.
    file_list = glob.glob(folder_calibration+"\*.h5")
    drivepsd2 = 0*(getdata(file_list[0])[0])
    xpsd2 = 0*(getdata(file_list[0])[0])
    for i in file_list:
        a = getdata(i)
        freq = a[0]
        drivepsd2 += a[2]
        xpsd2 += a[1]

    drivepsd2 = drivepsd2/len(file_list)
    xpsd2 = xpsd2/len(file_list)
    
    f0arg = np.argmax(drivepsd2)
    f0 = freq[f0arg]
    
    V2 = (np.sum(drivepsd2[f0arg-3:f0arg+3]))*(freq[f0arg+1] - freq[f0arg])*2.
    V = np.sqrt(V2)

    X2 = (np.sum(xpsd2[f0arg-3:f0arg+3]))*(freq[f0arg+1] - freq[f0arg])*2.
    X = np.sqrt(X2)

    # plt.figure()
    # plt.loglog(freq, drivepsd2)
    # plt.loglog(freq[f0arg], drivepsd2[f0arg], "ro")
    # plt.show()
    return [f0, f0arg, V, X, np.sqrt(xpsd2), freq]


def calibration(folder_calibration, folder_hp, file_high_pressure):# gives convertion between x displacement and field
    a = get_drive_and_motion(folder_calibration)
    f_field = a[0]

    V = a[2]

    f_resonance = find_resonance(folder_hp, file_high_pressure)[0]
    
    field_amp = 200.*V/d
    force_amp = (elec_charge)*field_amp
    acc_amp = force_amp/mass
    x_amp = acc_amp/( (2.*np.pi*f_resonance)**2 - (2.*np.pi*f_field)**2  )

    X = a[3]

    conversion_v_to_m = 1.0*x_amp/X

    return [conversion_v_to_m, f_field]

def plot_psd_meter(folder_meas, folder_hp, file_high_pressure, fmin, fmax):
    k = calibration(folder_calibration, folder_hp, file_high_pressure)[0]
    
    hp = find_resonance(folder_hp, file_high_pressure)
    xpsd_v_hp = hp[2]
    freqhp = hp[1]
    xpsd_m_hp = k*xpsd_v_hp

    meas = get_drive_and_motion(folder_meas)
    xpsd_meas_v = meas[4]
    xpsd_m_meas = k*xpsd_meas_v
    freq_meas = meas[5]

    ihpmin = np.where(freqhp >= fmin)[0][0]
    ihpmax = np.where(freqhp >= fmax)[0][0]
    
    imeasmin = np.where(freq_meas >= fmin)[0][0]
    imeasmax = np.where(freq_meas >= fmax)[0][0]

    plt.figure()
    plt.loglog(freqhp[ihpmin:ihpmax], xpsd_m_hp[ihpmin:ihpmax])
    plt.loglog(freq_meas[imeasmin:imeasmax], xpsd_m_meas[imeasmin:imeasmax])
    plt.xlabel("Freq [Hz]")
    plt.ylabel("m/sqrt(Hz)")
    plt.grid()

    return [freqhp, freq_meas, xpsd_m_hp, xpsd_m_meas]

def plot_psd_acc(folder_meas, folder_hp, file_high_pressure, fmin, fmax):
    a = plot_psd_meter(folder_meas, folder_hp, file_high_pressure, fmin, fmax)

    freqhp = a[0]
    freq_meas = a[1]
    xpsd_m_hp = a[2]
    xpsd_m_meas = a[3]

    ihpmin = np.where(freqhp >= fmin)[0][0]
    ihpmax = np.where(freqhp >= fmax)[0][0]
    
    imeasmin = np.where(freq_meas >= fmin)[0][0]
    imeasmax = np.where(freq_meas >= fmax)[0][0]

    f0 = find_resonance(folder_hp, file_high_pressure)[0]
    f = calibration(folder_calibration, folder_hp, file_high_pressure)[1]

    b = (2.*np.pi*f)**2

    acc_hp = b*a[2]
    acc_meas = b*a[3]

    plt.figure()
    plt.loglog(freqhp[ihpmin:ihpmax], acc_hp[ihpmin:ihpmax]/(9.8e-6))
    plt.loglog(freq_meas[imeasmin:imeasmax], acc_meas[imeasmin:imeasmax]/(9.8e-6))
    plt.xlabel("Freq [Hz]")
    plt.ylabel("ug/sqrt(Hz)")
    plt.grid()
    
    return [freqhp, freq_meas, xpsd_m_hp, xpsd_m_meas]



plot_psd_acc(folder_meas, folder_hp, file_high_pressure, freq_plot_min, freq_plot_max)
plt.show()

# print get_drive_and_motion(folder_calibration)
