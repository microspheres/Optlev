import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt

rho = 1800.0

R = 7.5*10**-6

M = (4./3.)*np.pi*(R**3)*rho

electron = 1.60218e-19

kb = 1.38e-23

acceleration_plot = False

no_sphere = True
pathno = [r"C:\data\20190326\15um_low532_50x\3\temp\no_sphere",]

distance = 0.02

NFFT = 2**16

path_calibration = r"C:\data\20190326\15um_low532_50x\3\calibration_1p"

path_list_temp = [r"C:\data\20190326\15um_low532_50x\3\temp\1", r"C:\data\20190326\15um_low532_50x\3\temp\2", r"C:\data\20190326\15um_low532_50x\3\temp\3", r"C:\data\20190326\15um_low532_50x\3\temp\4",r"C:\data\20190326\15um_low532_50x\3\temp\5", r"C:\data\20190326\15um_low532_50x\3\temp\6", r"C:\data\20190326\15um_low532_50x\3\temp\7", r"C:\data\20190326\15um_low532_50x\3\temp\8", r"C:\data\20190326\15um_low532_50x\3\temp\9", r"C:\data\20190326\15um_low532_50x\3\temp\10", r"C:\data\20190326\15um_low532_50x\3\temp\11trekoff", r"C:\data\20190326\15um_low532_50x\3\temp\12", r"C:\data\20190326\15um_low532_50x\3\temp\13", r"C:\data\20190326\15um_low532_50x\3\temp\14", r"C:\data\20190326\15um_low532_50x\3\temp\15", r"C:\data\20190326\15um_low532_50x\3\temp\16", ]

# path_list_temp = [r"C:\data\20190402\Trek_no_sphere_another_table_ON", r"C:\data\20190402\Trek_no_sphere_another_table_OFF"]

# path_list_temp = [r"C:\data\20190326\15um_low532_50x\4\temp\1", r"C:\data\20190326\15um_low532_50x\4\temp\2", r"C:\data\20190326\15um_low532_50x\4\temp\3", r"C:\data\20190326\15um_low532_50x\4\temp\4", r"C:\data\20190326\15um_low532_50x\4\temp\5", r"C:\data\20190326\15um_low532_50x\4\temp\6", r"C:\data\20190326\15um_low532_50x\4\temp\nosphere"]

# path_list_temp = [r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\1", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\2", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\3", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\4", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\5", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\6", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\7", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\8", r"C:\data\20190326\15um_low532_50x\8\1e\differentdgx\9", ]
 
path_high_pressure_nofb= r"C:\data\20190326\15um_low532_50x\3"
file_high_pressure_nofb = "2mbar_yzcool.h5"

f_start = 60. # for the fit
f_end = 120. # for the fit

delta = 1e-2
fq = np.arange(f_start, f_end, delta)

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

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT)
        fieldpsd, freqs = matplotlib.mlab.psd((dat[:, 3]-numpy.mean(dat[:, 3])), Fs = Fs, NFFT = NFFT)
        
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
        dgx = info[2][0]
        Xpsd = np.zeros(len(freq))
        fieldpsd = np.zeros(len(freq))
        aux = get_files_path(path)
        for i in aux:
                a = getdata(i)
                Xpsd += a[1]
                fieldpsd += a[4]
                p = a[3]
        Xpsd = Xpsd/len(aux)
        fieldpsd = fieldpsd/len(aux)
        return [Xpsd, dgx, p, fieldpsd, freq]

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
    v = 200.*np.sum(a[3][pos-3:pos+3])
    v_amp = np.sqrt(v)/np.pi
    E_amp = v/distance
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
        sen = np.sum(a[0][pos-3:pos+3])
        sen_amp = np.sqrt(sen)/np.pi
        return sen_amp
        

def psd(f, A, f0, gamma):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    s1 = 2.*A*(gamma*(w0**2))
    s2 = 1.*(w0**2)*((w0**2 - w**2)**2 + (gamma*w)**2)
    s = np.sqrt(s1/s2)
    return A*s

def fit_high_pressure_no_fb(path_hp, file_hp):
        a = get_high_pressure_psd(path_hp, file_hp)
        freq = a[0]
        xpsd = np.sqrt(a[1])
        fit_points1 = np.logical_and(freq > f_start, freq < 59.0)
        fit_points2 = np.logical_and(freq > 61.0, freq < 119.0)
        fit_points3 = np.logical_and(freq > 121.0, freq < 179.0)
        fit_points4 = np.logical_and(freq > 181.0, freq < f_end)
        fit_points_new = fit_points1 + fit_points2 + fit_points3 + fit_points4
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
       temp = tempaux/kb
       return [temp, dgx, popt, freq, xpsd]


def temp_path_list(pathlist, path_hp, file_hp, pathcharge, pathno, acc):
        T = []
        Dgx = []
        f = fq
        hp = fit_high_pressure_no_fb(path_hp, file_hp)
        Conv = convert_sensor_meter(pathcharge, path_hp, file_hp)
        plt.figure()
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("m/sqrt(Hz)")
        plt.loglog(hp[1], Conv*hp[3])
        labelhp = " $\Gamma/2\Pi$ = " + str("%.1E" % hp[0][2]) + " Hz"
        plt.loglog(hp[2], Conv*psd(hp[2], *hp[0]), "k",label = labelhp)

        if no_sphere:
                ns = tempeture_path(pathno[0], path_hp, file_hp, pathcharge)
                plt.loglog(ns[3], Conv*ns[4], label = "No Sphere")
                
        for i in pathlist:
                a = tempeture_path(i, path_hp, file_hp, pathcharge)
                dgx = a[1]
                t = a[0]
                T.append(t)
                Dgx.append(dgx)

                label = " $\Gamma/2\Pi$ = " + str("%.1E" % a[2][2]) + " Hz"
                plt.loglog(a[3], Conv*a[4])
                plt.loglog(f, Conv*psd(f, *a[2]), label = label)
                plt.xlim(1, 900)
                plt.ylim(1e-13, 1e-7)
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
        plt.loglog(Dgx, 1e6*np.array(T), "ro")
        plt.xlabel("Dgx")
        plt.ylabel("Temp [uK]")
        plt.legend(loc=3)
        plt.grid()
        plt.tight_layout(pad = 0)
        
        return [T, Dgx]



t2 = temp_path_list(path_list_temp, path_high_pressure_nofb, file_high_pressure_nofb, path_calibration, pathno, acceleration_plot)

    
plt.show()
