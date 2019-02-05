import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt

path300k = r"C:\data\20190202\15um\4"
name300k = r"2mbar_yzcool.h5"

no_sphere = True
if no_sphere:
        pathno = [r"C:\data\20190202\15um\4\PID\laseroffx"]

f_start = 50. # for the fit
f_end = 300. # for the fit

NFFT = 2**17

kb = 1.38*10**-23

mass = 2.*2.3*10**-26

vis = 18.54*10**-6

rho = 1800

R = 7.0*10**-6

M = (4./3.)*np.pi*(R**3)*rho

press = 240.

temp = 300


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
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT) 

	return [freqs, xpsd, PID]

data = getdata(os.path.join(path300k, name300k))

fit_points = np.logical_and(data[0] > f_start, data[0] < f_end)

fit_points1 = np.logical_and(data[0] > f_start, data[0] < 59.)
fit_points2 = np.logical_and(data[0] > 61. , data[0] < 106.)
fit_points3 = np.logical_and(data[0] > 107. , data[0] < 119.)
fit_points4 = np.logical_and(data[0] > 121. , data[0] < 130.)
fit_points_new = fit_points1+fit_points2 + fit_points3 + fit_points4

####################

def mean_free_path(vis, press, temp, mass):
    L1 = vis/press
    L2 = np.sqrt( np.pi*kb*temp/(2*mass) )
    return L1*L2

def Kn(vis, press, temp, mass, R):
    L = mean_free_path(vis, press, temp, mass)
    return L/R

def Gamma(vis, press, temp, mass, R, M):
    A = (6.0*np.pi*vis*R/M)
    B = 0.619/(0.619 + Kn(vis, press, temp, mass, R))
    C = (1. + 0.31*Kn(vis, press, temp, mass, R)/(0.785 + 1.152*Kn(vis, press, temp, mass, R)) )
    return A*B*C

def psd(f, A, f0, gamma):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    s1 = 2.*kb*temp*(gamma*(w0**2))
    s2 = 1.*M*(w0**2)*((w0**2 - w**2)**2 + (gamma*w)**2)
    s = np.sqrt(s1/s2)
    return A*s

gamma = Gamma(vis, press, temp, mass, R, M)

px, cx = opt.curve_fit(psd, data[0][fit_points_new], np.sqrt(data[1][fit_points_new]), p0 = [1e6, 100, gamma] )

f = np.arange(f_start, 130., 1e-2)

############################ folder with temperatures
path_list = [r"C:\data\20190202\15um\4\PID\full1", r"C:\data\20190202\15um\4\PID\full2", r"C:\data\20190202\15um\4\PID\full3",r"C:\data\20190202\15um\4\PID\full4", r"C:\data\20190202\15um\4\PID\full5", r"C:\data\20190202\15um\4\PID\full6", r"C:\data\20190202\15um\4\PID\full7", r"C:\data\20190202\15um\4\PID\full8", r"C:\data\20190202\15um\4\PID\full9", r"C:\data\20190202\15um\4\PID\full10", r"C:\data\20190202\15um\4\PID\full11", r"C:\data\20190202\15um\4\PID\full12"]

def get_files_path(path):
        file_list = glob.glob(path+"\*.h5")
        return file_list

print get_files_path(r"C:\data\20190202\15um\2\PID\full1")[0]

def get_data_path(path):
        info = getdata(get_files_path(path)[0])
        freq = info[0]
        dgx = info[2][0]
        Xpsd = np.zeros(len(freq))
        for i in get_files_path(path):
                a = getdata(i)
                Xpsd += a[1]
        return [Xpsd, dgx]
        

def psd_paths(pathlist):
        dgx = []
        xpsd = []
        for i in pathlist:
               a = get_data_path(i)
               dgx.append(a[1])
               xpsd.append(a[0])
        return [xpsd, dgx]

def fit_paths(pathlist):
        A = psd_paths(pathlist)[0]
        PX = []
        CX = []
        for i in range(len(A)):
                px_aux, cx_aux = opt.curve_fit(psd, data[0][fit_points_new], np.sqrt(A[i][fit_points_new]), p0 = [1e6, 110, gamma] )
                CX.append(cx_aux)
                PX.append(px_aux)
        return PX

def plot_all(pathlist):
        plt.figure()
        plt.loglog(data[0], np.sqrt(data[1])/px[0], label = "No feedback")
        plt.loglog(f, psd(f,*px)/px[0])
        aux = psd_paths(pathlist)
        aux2 = fit_paths(pathlist)
        for i in range(len(pathlist)):
                name = "dgx = " + str("%.1E" % aux[1][i]) + " $\Gamma$ = " + str("%.1E" % aux2[i][2]) + " Hz"
                plt.loglog(data[0], np.sqrt(aux[0][i])/px[0], label = name)
                plt.loglog(f, psd(f, *aux2[i])/px[0])

        plt.xlim(20,200)
        plt.ylim(2e-13,2e-8)

        if no_sphere:
                No = psd_paths(pathno)
                plt.loglog(data[0], np.sqrt(No[0][0])/px[0], label = "No sphere")
        
        plt.xlabel("Frequency[Hz]")
        plt.ylabel("$m/ \sqrt{Hz}$")
        plt.legend(loc=3)
        plt.grid()
        plt.tight_layout(pad = 0)

        ref = np.sum(psd(f,*px)/px[0])
        T = []
        dg = []
        for j in range(len(pathlist)):
                ref2 = np.sum(psd(f, *aux2[j])/px[0])
                temp = 300.*((ref2/ref)**2)
                temp = 1e6*temp
                T.append(temp)
                dg.append(aux[1][j])
        plt.figure()
        plt.loglog(dg, T, "ro")
        plt.xlabel("dgx")
        plt.ylabel("COMx Temp [$\mu$K]")
        plt.show()

plot_all(path_list)
