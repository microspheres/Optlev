import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt

plot = True
several_folders = False
if several_folders:
    folder_temp = r"C:\data\20191122\10um\2\temp_x9"
    plot = False

folder_calibration = r"C:\data\20191122\10um\2\calibration1p"

folder_meas = r"C:\data\20191122\10um\2\calibration1p"

folder_hp = r"C:\data\20191122\10um\2\1mbar"
file_high_pressure = r"1mbar_zcool.h5"

file_list = glob.glob(folder_calibration+"\*.h5")

NFFT = 2**18

drive_col = 3

#Diameter = 22.6e-6 #meters
Diameter = 10.0e-6
rho = 1800

d = 0.0029
#d = 0.002

number_of_charge = 1.0

elec_charge = number_of_charge*(1.60218e-19)


def mass(Diameter, rho):
    m = (4/3.)*(np.pi)*((Diameter/2)**3)*rho
    return m

mass = mass(Diameter, rho)
print "Mass = ", mass

freq_plot_min = 10.
freq_plot_max = 3000.

def getdata(fname, channelX, NFFT):
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
                print PID
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)

                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, channelX]-numpy.mean(dat[:, channelX]), Fs = Fs, NFFT = NFFT)
        drivepsd, freqs = matplotlib.mlab.psd(dat[:, drive_col]-numpy.mean(dat[:, drive_col]), Fs = Fs, NFFT = NFFT)

        # plt.figure()
        # plt.plot(dat[:, drive_col]-numpy.mean(dat[:, drive_col]))
        # plt.show()

        x = dat[:, bu.xi]-numpy.mean(dat[:, bu.xi])
        # v = np.gradient(x, 1./Fs)
        # a = np.gradient(v, 1./Fs)
        # apsd , freqs = matplotlib.mlab.psd(a, Fs = Fs, NFFT = NFFT)
        # plt.figure()
        # plt.loglog(freqs, np.sqrt(apsd))
        # plt.loglog(freqs, np.sqrt(xpsd))
        # plt.show()

	return [freqs, xpsd, drivepsd, PID[0]]

def psd(f, A, f0, gamma):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    gamma1 = 2.0*np.pi*gamma
    s1 = 1.*gamma
    s2 = 1.*((w0**2 - w**2)**2 + (gamma1*w)**2)
    s = np.sqrt(s1/s2)
    return A*s

def psdLP(f, A, f0, gamma):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    gamma1 = 2.0*np.pi*gamma
    s1 = 1.
    s2 = 1.*((w0**2 - w**2)**2 + (gamma1*w)**2)
    s = np.sqrt(s1/s2)
    return A*s

def find_resonance(folder_hp, file_high_pressure, channelX):
    a = getdata(os.path.join(folder_hp, file_high_pressure), channelX, 2**18)
    freq = a[0]
    xpsd = np.sqrt(a[1])
    xpsd2 = a[1]
    popt, pcov = opt.curve_fit(psd, freq[1000:2500], xpsd[1000:2500], p0 = [3000, 65, 20])
    # plt.figure()
    # plt.semilogy(freq[1000:2500], xpsd[1000:2500])
    # plt.semilogy(freq, psd(freq, *popt))
    # plt.xlim(10, 150)
    # plt.show()
    return [popt[1], freq, xpsd2]


def get_drive_and_motion(folder_calibration, channelX, NFFT): # return freq, freq_arg, amplitude of the drive and sphere x motion in the freq band.
    file_list = glob.glob(folder_calibration+"\*.h5")
    drivepsd2 = 0*(getdata(file_list[0], channelX, NFFT)[0])
    xpsd2 = 0*(getdata(file_list[0], channelX, NFFT)[0])
    for i in file_list:
        a = getdata(i, channelX, NFFT)
        freq = a[0]
        drivepsd2 += a[2]
        xpsd2 += a[1]
        xDg = a[3]

    drivepsd2 = drivepsd2/len(file_list)
    xpsd2 = xpsd2/len(file_list)
    
    f0arg = np.argmax(drivepsd2)
    f0 = freq[f0arg]
    
    V2 = (np.sum(drivepsd2[f0arg-3:f0arg+3]))*(freq[f0arg+1] - freq[f0arg])*2.

    X2 = (np.sum(xpsd2[f0arg-3:f0arg+3]))*(freq[f0arg+1] - freq[f0arg])*2.
    
    # plt.figure()
    # plt.loglog(freq, drivepsd2)
    # plt.loglog(freq[f0arg], drivepsd2[f0arg], "ro")
    # plt.show()
    return [f0, f0arg, V2, X2, xpsd2, freq, xDg]


def calibration(folder_calibration, folder_hp, file_high_pressure, channelX, NFFT):# gives convertion between x displacement and field
    a = get_drive_and_motion(folder_calibration, channelX, NFFT)
    f_field = a[0]
    f_arg_field = a[1]
    # f_resonance = find_resonance(folder_hp, file_high_pressure, channelX)[0]

    # find gamma at LP
    xpsd = np.sqrt(a[4])
    freq = a[5]
    fit_points1 = np.logical_and(freq > freq[1000], freq < 57.1)
    fit_points2 = np.logical_and(freq > 57.4, freq < 59)
    fit_points3 = np.logical_and(freq > 61, freq < freq[2000])
    fit = fit_points1 + fit_points2 + fit_points3
    popt_g, pcov_g = opt.curve_fit(psdLP, freq[fit], xpsd[fit], p0 = [0.01, 60, 1])
    f_resonance = popt_g[1]
    gamma_cali = np.abs(popt_g[2])

    print "gamma", gamma_cali
    plt.figure()
    plt.loglog(freq[1200:2000], xpsd[1200:2000])
    plt.loglog(freq, psdLP(freq, *popt_g))
    plt.show()

    V = np.sqrt(a[2])
    print "voltage for calibration Vpp = ", 2*V
    
    field_amp = 200.*V/d
    force_amp = (elec_charge)*field_amp
    acc_amp = force_amp/mass
    x_amp = acc_amp/np.sqrt( (( 2.*np.pi*f_resonance)**2 - (2.*np.pi*f_field)**2 )**2 + (2.*np.pi*f_field*2.*np.pi*gamma_cali)**2 )
    #x_amp = acc_amp/np.sqrt( (( 2.*np.pi*f_resonance)**2 - (2.*np.pi*f_field)**2 )**2 )
    # the zero above is because there is no difference for small derivative gain.
    x_amp2 = 0.5*(x_amp**2) # the 0.5 comes from the fft**2. To see that use parseval theorem in x_amp from and find that x_amp2 = 0.5*x_amp**2

    X2 = a[3]

    conversion_v2_to_m2 = 1.0*x_amp2/X2

    print "convertion_v2_to_m2 = ", conversion_v2_to_m2

    if channelX == 0:
        name = "in"
    if channelX == 4:
        name = "out"
    saveinfo = conversion_v2_to_m2
    savename = str(folder_meas) + "\\" + "v2tom2_" + name
    np.save(savename, saveinfo)

    return [conversion_v2_to_m2, f_field, f_arg_field, gamma_cali, f_resonance]

def plot_psd_meter(folder_meas, folder_hp, file_high_pressure, fmin, fmax, savename, channelX, NFFT):
    k, f_field, f_field_arg, gamma, fres = calibration(folder_calibration, folder_hp, file_high_pressure, channelX, NFFT)
    
    hp = find_resonance(folder_hp, file_high_pressure, channelX)
    xpsd2_v_hp = hp[2]
    freqhp = hp[1]
    xpsd2_m_hp = k*xpsd2_v_hp

    meas = get_drive_and_motion(folder_meas, channelX, NFFT)
    xpsd2_meas_v = meas[4]
    xpsd2_m_meas = k*xpsd2_meas_v
    freq_meas = meas[5]

    ihpmin = np.where(freqhp >= fmin)[0][0]
    ihpmax = np.where(freqhp >= fmax)[0][0]
    
    imeasmin = np.where(freq_meas >= fmin)[0][0]
    imeasmax = np.where(freq_meas >= fmax)[0][0]

    if plot:
        plt.figure()
        plt.loglog(freqhp[ihpmin:ihpmax], np.sqrt( xpsd2_m_hp[ihpmin:ihpmax]) )
        plt.loglog(freq_meas[imeasmin:imeasmax], np.sqrt( xpsd2_m_meas[imeasmin:imeasmax]) )
        plt.xlabel("Freq [Hz]")
        plt.ylabel("$m/\sqrt{Hz}$")
        plt.grid()

    xDg = meas[6]

    saveinfo = [freqhp, freq_meas, xpsd2_m_hp, xpsd2_m_meas, xpsd2_v_hp, xpsd2_meas_v, xDg, fres]
    savename = str(folder_meas)+ savename
    print savename
    np.save(savename, saveinfo)

    return [freqhp, freq_meas, xpsd2_m_hp, xpsd2_m_meas, f_field_arg, gamma, fres]

def plot_psd_acc(folder_meas, folder_hp, file_high_pressure, fmin, fmax, savename, channelX, NFFT):
    a = plot_psd_meter(folder_meas, folder_hp, file_high_pressure, fmin, fmax, savename, channelX, NFFT)

    freqhp = a[0]
    freq_meas = a[1]
    xpsd2_m_hp = a[2]
    xpsd2_m_meas = a[3]
    f_field_arg = a[4]
    gamma = a[5]
    f0 = a[6]

    ihpmin = np.where(freqhp >= fmin)[0][0]
    ihpmax = np.where(freqhp >= fmax)[0][0]
    
    imeasmin = np.where(freq_meas >= fmin)[0][0]
    imeasmax = np.where(freq_meas >= fmax)[0][0]

    # f0 = find_resonance(folder_hp, file_high_pressure, channelX)[0]
    # f = calibration(folder_calibration, folder_hp, file_high_pressure, channelX, NFFT)[1]

    # b = (2.*np.pi*f0)**2

    # acc2_hp = (b**2)*a[2]
    # acc2_meas = (b**2)*a[3]

    N = ( ((2.*np.pi)**2)*(f0**2 - freq_meas**2) )**2 + (((2.*np.pi)**2)*(gamma*freq_meas))**2
    N = np.sqrt(N)

    # acc2_hp = ((2.0*np.pi*(freqhp))**4)*a[2]
    # acc2_meas = ((2.0*np.pi*(freq_meas))**4)*a[3]

    acc2_hp = (N**2)*xpsd2_m_hp
    acc2_meas = (N**2)*xpsd2_m_meas

    if plot:
        plt.figure()
        plt.loglog(freqhp[ihpmin:ihpmax], np.sqrt(acc2_hp[ihpmin:ihpmax])/(9.8e-6))
        plt.loglog(freq_meas[imeasmin:imeasmax], np.sqrt(acc2_meas[imeasmin:imeasmax])/(9.8e-6))
        plt.loglog(freq_meas[f_field_arg], np.sqrt(acc2_meas[f_field_arg])/(9.8e-6), "ro")
        plt.xlabel("Freq [Hz]")
        plt.ylabel("sensitivity ug/sqrt(Hz)")
        plt.grid()


    if channelX == 0:
        name = "in"
    if channelX == 4:
        name = "out"
    savename = str(folder_meas) + "\\" + "variance_sensor_" + name
    saveinfo = acc2_meas[f_field_arg]
    np.save(savename, saveinfo)
    
    return [freqhp, freq_meas, xpsd2_m_hp, xpsd2_m_meas]


def get_folder_list(folder_temp):
    n = os.listdir(folder_temp)
    A = []
    for i in n:
        a = folder_temp + "\\" + i 
        A.append(a)
    return A

if several_folders:
    folder_list = get_folder_list(folder_temp)
    for i in folder_list:
        plot_psd_acc(i, folder_hp, file_high_pressure, freq_plot_min, freq_plot_max, r"\info_outloop", 4, NFFT)
        plot_psd_acc(i, folder_hp, file_high_pressure, freq_plot_min, freq_plot_max, r"\info_inloop", bu.xi, NFFT)
else:
    plot_psd_acc(folder_meas, folder_hp, file_high_pressure, freq_plot_min, freq_plot_max, r"\info_outloop", 4, NFFT)
    plot_psd_acc(folder_meas, folder_hp, file_high_pressure, freq_plot_min, freq_plot_max, r"\info_inloop", bu.xi, NFFT)
    plt.show()

  # print get_drive_and_motion(folder_calibration)
