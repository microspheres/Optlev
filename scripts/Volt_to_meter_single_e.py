import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt
from scipy.signal import butter, lfilter, filtfilt

order = 2
def butter_lowpass(lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a


def butter_lowpass_filter(data, lowcut, fs, order):
    b, a = butter_lowpass(lowcut, fs, order)
    y = filtfilt(b, a, data)
    return y

rho = 1800.0

R = 7.5*10**-6

M = (4./3.)*np.pi*(R**3)*rho
dR = (2.7/2)*10**-6
dM = M*np.sqrt( (dR/R)**2 )

electron = 1.60218e-19

kb = 1.38e-23

acceleration_plot = False

no_sphere = False
pathno = [r"C:\data\20190408\15um\3\temp\no_sphere",]

distance = 0.0105
distance_error = 1e-4

NFFT = 2**15

path_calibration = r"C:\data\201904011\15um\beforeFB\6\calibration1e"

# path_list_temp = [r"C:\data\20190326\15um_low532_50x\3\temp\1", r"C:\data\20190326\15um_low532_50x\3\temp\2", r"C:\data\20190326\15um_low532_50x\3\temp\3", r"C:\data\20190326\15um_low532_50x\3\temp\4",r"C:\data\20190326\15um_low532_50x\3\temp\5", r"C:\data\20190326\15um_low532_50x\3\temp\6", r"C:\data\20190326\15um_low532_50x\3\temp\7", r"C:\data\20190326\15um_low532_50x\3\temp\8", r"C:\data\20190326\15um_low532_50x\3\temp\9", r"C:\data\20190326\15um_low532_50x\3\temp\10", r"C:\data\20190326\15um_low532_50x\3\temp\11trekoff", r"C:\data\20190326\15um_low532_50x\3\temp\12", r"C:\data\20190326\15um_low532_50x\3\temp\13", r"C:\data\20190326\15um_low532_50x\3\temp\14", r"C:\data\20190326\15um_low532_50x\3\temp\15", r"C:\data\20190326\15um_low532_50x\3\temp\16", ]

# path_list_temp = [r"C:\data\20190402\Trek_no_sphere_another_table_ON", r"C:\data\20190402\Trek_no_sphere_another_table_OFF"]

# path_list_temp = [r"C:\data\20190326\15um_low532_50x\4\temp\1", r"C:\data\20190326\15um_low532_50x\4\temp\2", r"C:\data\20190326\15um_low532_50x\4\temp\3", r"C:\data\20190326\15um_low532_50x\4\temp\4", r"C:\data\20190326\15um_low532_50x\4\temp\5", r"C:\data\20190326\15um_low532_50x\4\temp\6", r"C:\data\20190326\15um_low532_50x\4\temp\nosphere"]

# path_list_temp = [r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\1", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\2", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\3", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\4", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\5", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\6", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\7", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\8", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\9", ]

# path_list_temp = [r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\temp\1", r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\temp\2", r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\temp\3", r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\temp\4", r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\temp\5", r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\temp\6", r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\temp\7",r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\temp\8", r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\temp\9", r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\temp\10", r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\temp\11", r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\temp\12"]

path_list_temp = [r"C:\data\20190408\15um\3\temp\1", r"C:\data\20190408\15um\3\temp\2", r"C:\data\20190408\15um\3\temp\3", r"C:\data\20190408\15um\3\temp\4", r"C:\data\20190408\15um\3\temp\5", r"C:\data\20190408\15um\3\temp\6", r"C:\data\20190408\15um\3\temp\7", r"C:\data\20190408\15um\3\temp\8", r"C:\data\20190408\15um\3\temp\9", r"C:\data\20190408\15um\3\temp\10", r"C:\data\20190408\15um\3\temp\11", r"C:\data\20190408\15um\3\temp\12", r"C:\data\20190408\15um\3\temp\13", r"C:\data\20190408\15um\3\temp\14", r"C:\data\20190408\15um\3\temp\HP",]

# path_list_temp = [r"C:\data\20190408\15um\3\temp\HP", r"C:\data\20190408\15um\3\temp\12", r"C:\data\20190408\15um\3\temp\13", r"C:\data\20190408\15um\3\temp\6", r"C:\data\20190408\15um\3\temp\3", r"C:\data\20190408\15um\3\temp\1",]

path_list_temp = [r"C:\data\20190408\15um\5\temp\1",r"C:\data\20190408\15um\5\temp\2", r"C:\data\20190408\15um\5\temp\3", r"C:\data\20190408\15um\5\temp\4", r"C:\data\20190408\15um\5\temp\5", r"C:\data\20190408\15um\5\temp\6", r"C:\data\20190408\15um\5\temp\7", r"C:\data\20190408\15um\5\temp\8", r"C:\data\20190408\15um\5\temp\9", r"C:\data\20190408\15um\5\temp\10", r"C:\data\20190408\15um\5\temp\11", r"C:\data\20190408\15um\5\temp\12", r"C:\data\20190408\15um\5\temp\no_sphere"]

path_list_temp = [r"C:\data\201904011\15um\beforeFB\5\temp\HP", r"C:\data\201904011\15um\beforeFB\5\temp\1", r"C:\data\201904011\15um\beforeFB\5\temp\2", r"C:\data\201904011\15um\beforeFB\5\temp\8", r"C:\data\201904011\15um\beforeFB\5\temp\10", r"C:\data\201904011\15um\beforeFB\5\temp\16", r"C:\data\201904011\15um\beforeFB\5\temp\tranfer_BF",]

path_list_temp = [r"C:\data\201904011\15um\beforeFB\5\temp\tranfer_BF", ]

path_list_temp = [r"C:\data\201904011\15um\beforeFB\6\temp\samegain\deflection2"]

path_list_temp = [r"C:\data\20190509_electricbackground"]

path_high_pressure_nofb= r"C:\data\201904011\15um\beforeFB\6"
file_high_pressure_nofb = "2mbar_yzcool.h5"

f_start = 60. # for the fit
f_end = 130. # for the fit

f_start = 40. # for the fit
f_end = 100. # for the fit

f_start = 50. # for the fit
f_end = 120. # for the fit

delta = 1e-2
fq = np.arange(f_start, f_end, delta)

csd_boolean = True 

nice_plot = False # plot the HP inside the pathlist and Not the one outside
plot_fit_HP = False
plot_fit_LP = True



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
                PID = dset.attrs['PID']
                press = dset.attrs['pressures']
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

        # d = butter_lowpass_filter(dat[:,4], 20., Fs, 4)
	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi] - numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT)
        fieldpsd, freqs = matplotlib.mlab.psd((dat[:, 3]-numpy.mean(dat[:, 3])), Fs = Fs, NFFT = NFFT)
        if csd_boolean:
                x = dat[:, bu.xi]-numpy.mean(dat[:, bu.xi])
                xb = dat[:, 4]/(dat[:, 6]+1) - numpy.mean(dat[:, 4]/(dat[:, 6]+1))
                bfpsd, freqs = matplotlib.mlab.psd(xb, Fs = Fs, NFFT = NFFT)

                mean = 0
                std = 1e-6
                num_samples = len(dat[:,4])
                samples = np.array(numpy.random.normal(mean, std, size=num_samples))

                freq, csd = sp.csd(xb, x, fs = Fs, nperseg = NFFT)


                # csd, freqs = matplotlib.mlab.cohere(x, xb, Fs = Fs, NFFT = NFFT)

                return [freqs, xpsd, PID, press[0], fieldpsd, csd, bfpsd]
        else:
                return [freqs, xpsd, PID, press[0], fieldpsd]

def get_high_pressure_psd(path_hp, file_hp):
        a = getdata(os.path.join(path_hp, file_hp))
        freq = a[0]
        xpsd = a[1]
        return [freq, xpsd]

    
def get_files_path(path):
        file_list = glob.glob(path+"\*.h5")
        return file_list


def get_data_path(path): # PSD output is unit square, V**2/Hz : it assumes that within the folder, Dgx is the same.
        info = getdata(get_files_path(path)[0])
        freq = info[0]
        dfreq = freq[1] - freq[0]
        dgx = info[2][0]
        Xpsd = np.zeros(len(freq))
        fieldpsd = np.zeros(len(freq))
        aux = get_files_path(path)
        csd = np.zeros(len(freq), dtype=complex)
        bfpsd = np.zeros(len(freq))
        for i in aux:
                a = getdata(i)
                Xpsd += a[1]
                fieldpsd += a[4]
                p = a[3]
                if csd_boolean:
                        csd += a[5]
                        bfpsd += a[6]
        Xpsd = Xpsd/len(aux)
        fieldpsd = fieldpsd/len(aux)
        csd = np.abs(csd)/len(aux)
        bfpsd = bfpsd/len(aux)
        return [Xpsd, dgx, p, fieldpsd, freq, dfreq, csd, bfpsd]


def plot_psd(path):
    a = get_data_path(path)
    freq = a[4]
    plt.figure()
    plt.loglog(freq, a[0])
    plt.loglog(freq, a[3])
    return "hi!"

def findAC_peak(path):
    a = get_data_path(path)
    freq = a[4]
    pos = np.argmax(a[3])
    return [pos, freq[pos]]

def get_field(path):
    a = get_data_path(path)
    pos = findAC_peak(path)[0]
    v = np.sum(a[3][pos-3:pos+3])*a[5]
    v_amp = 200.0*np.sqrt(v)*np.sqrt(2.)
    E_amp = v_amp/distance
    return [v_amp, E_amp]



def force1e(path): #gives force of 1e of charge
    E = get_field(path)[1]
    F = E*electron
    return F

def acc(path): # gives the acc of 1e of charge
    F = force1e(path)
    acc = F/M
    return acc

def get_sensor_motion_1e(path):
        pos = findAC_peak(path)[0]
        a = get_data_path(path)
        sen = np.sum(a[0][pos-3:pos+3])*a[5]
        sen_amp = np.sqrt(sen)*np.sqrt(2.)
        return sen_amp
        

def psd(f, A, f0, gamma):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    gamma1 = 2.0*np.pi*gamma
    s1 = 2.*A*(gamma1*(w0**2))
    s2 = 1.*(w0**2)*((w0**2 - w**2)**2 + (gamma1*w)**2)
    s = np.sqrt(s1/s2)
    return A*s

def fit_high_pressure_no_fb(path_hp, file_hp):
        a = get_high_pressure_psd(path_hp, file_hp)
        freq = a[0]
        xpsd = np.sqrt(a[1])
        fit_points1 = np.logical_and(freq > f_start, freq < 59.0)
        fit_points2 = np.logical_and(freq > 61.0, freq < 119.0)
        fit_points3 = np.logical_and(freq > 121.0, freq < 122.0)
        fit_points4 = np.logical_and(freq > 123.3, freq < 144.8)
        fit_points5 = np.logical_and(freq > 145.9, freq < 179.0)
        fit_points6 = np.logical_and(freq > 181.0, freq < f_end)
        fit_points_new = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5
        p0 = [0.1, 90, 100.]
        popt, pcov = opt.curve_fit(psd, freq[fit_points_new], xpsd[fit_points_new], p0 = p0)
        freqplot = fq
        # plt.figure()
        # plt.loglog(freq, xpsd)
        # plt.loglog(freqplot, psd(freqplot, *popt))
        return [popt, freq, freqplot, xpsd]
        

def convert_sensor_meter(path, path_hp, file_hp): # given that the feedback barelly affects the motion due to the ac field
        sen_amp = get_sensor_motion_1e(path)
        acc1e = acc(path)
        f0 = fit_high_pressure_no_fb(path_hp, file_hp)[0][1]
        motiontheo = 1.0*acc1e/((2.0*np.pi*f0)**2)
        C = 1.0*motiontheo/sen_amp
        return C


def tempeture_path(path, path_hp, file_hp, pathcharge):
       a = get_data_path(path)
       xpsd = np.sqrt(a[0])
       dgx = a[1]
       freq = a[4]
       Conv = convert_sensor_meter(pathcharge, path_hp, file_hp)
       b = fit_high_pressure_no_fb(path_hp, file_hp)[0]
       f0 = b[1]

       fit_points1 = np.logical_and(freq > f_start, freq < 59.6)
       fit_points2 = np.logical_and(freq > 60.6 , freq < 65.0)
       fit_points3 = np.logical_and(freq > 65.7 , freq < 95.5)
       fit_points4 = np.logical_and(freq > 96.2 , freq < 119.0)
       
       fit_points5 = np.logical_and(freq > 121 , freq < f_end)
       fit_points_new = fit_points1+fit_points2+fit_points3+fit_points4
       p0 = [1e-1, np.abs(f0), 100.]
       popt, pcov = opt.curve_fit(psd, freq[fit_points_new], xpsd[fit_points_new], p0 = p0)
       
       f = fq
       aux = (2.*np.pi*np.abs(f0))*Conv*psd(f, *popt)
       tempaux = np.sum(aux**2)*delta
       tempaux = 0.5*M*tempaux
       temp = 2.*tempaux/kb # factor 2 is to account the spring energy (equipartition theo)

       if csd_boolean:
               csd = np.sqrt(a[6])
               bfpsd = np.sqrt(a[7])
               return [temp, dgx, popt, freq, xpsd, csd, bfpsd]
       else:
               return [temp, dgx, popt, freq, xpsd]


def temp_path_list(pathlist, path_hp, file_hp, pathcharge, pathno, acc):
        T = []
        dT = []
        Dgx = []
        f = fq
        hp = fit_high_pressure_no_fb(path_hp, file_hp)
        Conv = convert_sensor_meter(pathcharge, path_hp, file_hp)
        print Conv
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        plt.figure()
        plt.xlabel("Frequency $[Hz]$", fontsize = 15)
        plt.ylabel(r"$\sqrt{S} \ [ m/\sqrt{Hz}]$", fontsize = 15)
        if not nice_plot:
                plt.plot(hp[1], Conv*hp[3])
                labelhp = " $\Gamma/2\Pi$ = " + str("%.1E" % hp[0][2]) + " Hz"
                if plot_fit_HP:
                        plt.plot(hp[2], Conv*psd(hp[2], *hp[0]), "--k")

        if no_sphere:
                ns = tempeture_path(pathno[0], path_hp, file_hp, pathcharge)
                plt.plot(ns[3], Conv*ns[4], "-r",label = "No Sphere")
                
        for i in pathlist:
                a = tempeture_path(i, path_hp, file_hp, pathcharge)
                dgx = a[1]
                t = a[0]
                dt = t*np.sqrt(( (dM/M)**2 + (2.*distance_error/distance)**2 ))
                T.append(t)
                dT.append(dt)
                Dgx.append(dgx)
                print "resonace freq =", a[2][1]
                label = " $\Gamma/2\Pi$ = " + str("%.1E" % a[2][2]) + " Hz"
                plt.plot(a[3], Conv*a[4])
                if plot_fit_LP:
                        plt.loglog(f, Conv*psd(f, *a[2]), "--k")
                plt.xlim(1, 2000)
                plt.ylim(1e-12, 3e-8)

        plt.legend(loc=3)
        plt.grid()
        plt.tight_layout(pad = 0)

        if acc: # only to know the acc sensitivity
                C = (2.0*np.pi*hp[0][1])**2
                plt.figure()
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("m/s**2/sqrt(Hz)")
                plt.loglog(hp[1], C*Conv*hp[3])
                plt.loglog(hp[2], C*Conv*psd(hp[2], *hp[0]))
                if no_sphere:
                        ns = tempeture_path(pathno[0], path_hp, file_hp, pathcharge)
                        plt.loglog(ns[3], C*Conv*ns[4], label = "No Sphere")
                for i in pathlist:
                        a = tempeture_path(i, path_hp, file_hp, pathcharge)
                
                        plt.loglog(a[3], C*Conv*a[4])
                        plt.loglog(f, C*Conv*psd(f, *a[2]))
                plt.xlim(1, 500)
                # plt.ylim(1e-13, 1e-7)
                plt.legend(loc=3)
                plt.grid()
                plt.tight_layout(pad = 0)
                
                
        plt.figure()
        plt.errorbar(Dgx, 1e6*np.array(T), yerr = 1e6*np.array(dT), fmt = "ro")
        for i in np.array(Dgx):
                log = True
                if i == 0:
                        log = False
        if log:
                plt.xscale("log")
                plt.yscale("log")
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        plt.xlabel("Feedback Gain [Arb. Units]", fontsize = 15)
        plt.ylabel("COM Temperature [$\mu $K]", fontsize = 15)
        plt.legend(loc=3)
        plt.grid()
        plt.tight_layout(pad = 0)
        
        return [T, dT, Dgx]

def plot_csd(pathlist, path_hp, file_hp, pathcharge):
        plt.figure()
        for i in pathlist:
                a = tempeture_path(i, path_hp, file_hp, pathcharge)
                xpsd = a[4]
                freqs = a[3]
                csd = a[5]
                bfpsd = a[6]
                plt.loglog(freqs, xpsd, label = "Xpsd")
                plt.loglog(freqs, csd, "--", label = "CSD")
                plt.loglog(freqs, bfpsd, label = "BF_psd")
                norm = (csd**2)/(xpsd*bfpsd)
                plt.loglog(freqs, np.sqrt(norm), label = "Norm CSD")
                plt.xlim(1, 200)
                plt.legend(loc=3)
                plt.grid()
                plt.tight_layout(pad = 0)
        return [freqs, xpsd, csd]



t2 = temp_path_list(path_list_temp, path_high_pressure_nofb, file_high_pressure_nofb, path_calibration, pathno, acceleration_plot)

if csd_boolean:
        plot_csd(path_list_temp, path_high_pressure_nofb, file_high_pressure_nofb, path_calibration)
    
plt.show()
