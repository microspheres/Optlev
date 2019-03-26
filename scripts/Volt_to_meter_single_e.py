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

path300k = r"C:\data\20190304\15um_low532\6\2mbar"

no_sphere = True
if no_sphere:
        pathno = [r"C:\data\20190304\15um_low532\6"]
        fileno = r"nosphere.h5"

distance = 0.012

f_start = 50. # for the fit
f_end = 200. # for the fit

NFFT = 2**16

path_list = [r"C:\data\20190304\15um_low532\6\PID\gx1", r"C:\data\20190304\15um_low532\6\PID\gx2", r"C:\data\20190304\15um_low532\6\PID\gx3", r"C:\data\20190304\15um_low532\6\PID\gx4", r"C:\data\20190304\15um_low532\6\PID\gx5", r"C:\data\20190304\15um_low532\6\PID\gx6",]


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


    
def get_files_path(path):
        file_list = glob.glob(path+"\*.h5")
        return file_list


def get_data_path(path):
        info = getdata(get_files_path(path)[0])
        freq = info[0]
        dgx = info[2][0]
        Xpsd = np.zeros(len(freq))
        fieldpsd = np.zeros(len(freq))
        for i in get_files_path(path):
                a = getdata(i)
                Xpsd += a[1]
                fieldpsd += a[4]
                p = a[3]
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

def psd(f, A, f0, gamma):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    s1 = 2.*A*(gamma*(w0**2))
    s2 = 1.*(w0**2)*((w0**2 - w**2)**2 + (gamma*w)**2)
    s = np.sqrt(s1/s2)
    return A*s

def get_resonance_damp_dgx(path, conversion):
    aux = get_data_path(path)
    xpsd = np.sqrt(aux[0])
    freq = aux[4]
    dgx = aux[1]
    
    fit_points1 = np.logical_and(freq > f_start, freq < 59.)
    fit_points2 = np.logical_and(freq > 61. , freq < 119)
    fit_points3 = np.logical_and(freq > 121 , freq < f_end)
    fit_points = fit_points1+fit_points2+fit_points3


    f = np.arange(f_start, f_end, 1)
    p0 = [0.1, 80., 160.]
    px, cx = opt.curve_fit(psd, freq[fit_points], xpsd[fit_points], p0 = p0)

    if True:
        plt.figure()
        plt.loglog(freq, conversion*xpsd)
        plt.loglog(f, conversion*psd(f, *px))

    return [px[1], px[2], dgx]


def get_param_lp(file_list, conversion): # file list composed of files with different dgx
    Damp = []
    dgx = []
    f0 = []

    for i in file_list:
        aux = get_resonance_damp_dgx(i, conversion)
        Damp.append(aux[1])
        dgx.append(aux[2])
        f0.append(aux[0])

    return [Damp, f0, dgx]
        
        

f = get_resonance_damp_dgx(path300k, 1.)
f2 = get_param_lp(path_list, 1.)
print f






    
# acc = acc(path_1e)
# aux = findAC_peak(path_1e)
# peakpos = aux[0]
# freq_0 = aux[1]
# aux2 = get_data_path(path_1e)
# freq = aux2[4]
# xpsd_volt = aux2[0]
# x_amp_volt = np.sum(xpsd_volt[peakpos - 3:peakpos +3])
# x_amp_volt = np.sqrt(x_amp_volt) # not dividing by pi because I am comparing to the conversion made above.

# print x_amp_volt

# Z = np.sqrt( (630.)**2 + (1/((2.*np.pi*freq_0)**2))*((2.*np.pi*px[1])**2 - (2.*np.pi*freq_0)**2)**2  )

# x_m = acc/(Z*(2.*np.pi*freq_0))

# convertion_v_to_m = x_amp_volt/x_m

# print convertion_v_to_m

# plt.figure()
# plt.loglog(data[0], np.sqrt(data[1])/px[0], label = "psd conversion")
# plt.loglog(data[0], np.sqrt(data[1])/convertion_v_to_m, label = "1e conversion")
# plt.legend()

    

# path_1e = r"C:\data\20190304\15um_low532\6\1electron" 
# plot_psd(path_1e)

# a = findAC_peak(path_1e)
# print a

# b = get_field(path_1e)
# print b

# f = force1e(path_1e)
# print f

# a = acc(path_1e)
# print a

plt.show()
    
        

    
