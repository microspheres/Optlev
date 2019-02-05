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
        fileno = r"pd_xyzcool.h5"

f_start = 50. # for the fit
f_end = 110. # for the fit

NFFT = 2**17

kb = 1.38*10**-23

mass = 2.*2.3*10**-26

vis = 18.54*10**-6
vis_hidrogen = 1.37e-5

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
                press = dset.attrs['pressures']
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.xi]-numpy.mean(dat[:, bu.xi]), Fs = Fs, NFFT = NFFT) 

	return [freqs, xpsd, PID, press[0]]

data = getdata(os.path.join(path300k, name300k))

fit_points = np.logical_and(data[0] > f_start, data[0] < f_end)

fit_points1 = np.logical_and(data[0] > f_start, data[0] < 59.6)
fit_points2 = np.logical_and(data[0] > 60.6 , data[0] < f_end)
fit_points_new = fit_points1+fit_points2

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

f = np.arange(f_start, f_end, 1e-2)

############################ folder with temperatures
path_list = [r"C:\data\20190202\15um\4\PID\full1", r"C:\data\20190202\15um\4\PID\full2", r"C:\data\20190202\15um\4\PID\full3",r"C:\data\20190202\15um\4\PID\full4", r"C:\data\20190202\15um\4\PID\full5", r"C:\data\20190202\15um\4\PID\full6", r"C:\data\20190202\15um\4\PID\full7", r"C:\data\20190202\15um\4\PID\full8", r"C:\data\20190202\15um\4\PID\full9", r"C:\data\20190202\15um\4\PID\full10", r"C:\data\20190202\15um\4\PID\full11", r"C:\data\20190202\15um\4\PID\full12"]


######### Gamma low pressure
PP = getdata(glob.glob(path_list[0]+"\*.h5")[0])[3]
G = Gamma(vis_hidrogen, PP, temp, mass, R, M)
G = 100.*G # to Pa

####################


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
                px_aux, cx_aux = opt.curve_fit(psd, data[0][fit_points_new], np.sqrt(A[i][fit_points_new]), p0 = [1e6, abs(px[1]), gamma], bounds = ([1e2, abs(px[1])-10., 0.001*gamma], [1e10, abs(px[1])+10., 200.*gamma]) )
                CX.append(cx_aux)
                PX.append(px_aux)
        return [PX, CX]

def plot_all(pathlist):
        plt.figure()
        name_hp = "dgx = " + str("%.1E" % data[2][0]) + " $\Gamma$ = " + str("%.1E" % px[2]) + " Hz" + " $f_0$ = " + str("%.1E" % abs(px[1])) + " $\pm$ " + str("%.0E" % np.sqrt(abs(cx[1][1])))  +  " Hz"
        plt.loglog(data[0], np.sqrt(data[1])/px[0], label = name_hp)
        plt.loglog(f, psd(f,*px)/px[0])
        aux = psd_paths(pathlist)
        aux2 = fit_paths(pathlist)
        for i in range(len(pathlist)):
                if i == 0 or  i == 11:
                        name = "dgx = " + str("%.1E" % aux[1][i]) + " $\Gamma$ = " + str("%.1E" % aux2[0][i][2]) + " Hz" + " $f_0$ = " + str("%.1E" % aux2[0][i][1]) + " $\pm$ " + str("%.0E" % np.sqrt(aux2[1][i][1][1])) + " Hz"
                        plt.loglog(data[0], np.sqrt(aux[0][i])/px[0], label = name)
                        plt.loglog(f, psd(f, *aux2[0][i])/px[0])

        plt.xlim(20,200)
        plt.ylim(2e-13,2e-8)

        if no_sphere:
                No = getdata(os.path.join(pathno[0], fileno))
                plt.loglog(No[0][fit_points_new], np.sqrt(No[1][fit_points_new])/px[0], label = "No sphere")
                pno, cno = opt.curve_fit(psd, No[0][fit_points_new], np.sqrt(No[1][fit_points_new]), p0 = [1e6, abs(px[1]), gamma], bounds = ([1e2, abs(px[1])-15., 0.001*gamma], [1e10, abs(px[1])+15., 200.*gamma]) )
                plt.loglog(f, psd(f, *pno)/px[0])
                
        
        plt.xlabel("Frequency[Hz]")
        plt.ylabel("$m/ \sqrt{Hz}$")
        plt.legend(loc=3)
        plt.grid()
        plt.tight_layout(pad = 0)

        #####temp

        ref = np.sum(psd(f,*px)/px[0])
        T = []
        dg = []
        for j in range(len(pathlist)):
                ref2 = np.sum(psd(f, *aux2[0][j])/px[0])
                temp = 300.*((ref2/ref)**2)
                temp = 1e6*temp
                T.append(temp)
                dg.append(aux[1][j])
        if no_sphere:
                t_no = 300.*(np.sum((psd(f, *pno)/px[0]))/(ref))**2
                aa = np.arange(0, 2, 0.01)
                tt = t_no*np.ones(len(aa))
        plt.figure()
        plt.loglog(dg, T, "ro")
        if no_sphere:
                plt.loglog(aa, 1e6*tt, "b", alpha = 0.5, label = "System Noise")
                plt.fill_between(aa, 1e6*tt, facecolor='blue', alpha=0.5)
        plt.xlabel("dgx")
        plt.ylabel("COMx Temp [$\mu$K]")
        plt.xlim(0.01,1)
        plt.legend(loc=3)
        plt.grid()
        plt.tight_layout(pad = 0)

        ########

        plt.figure()
        for i in range(len(pathlist)):
                plt.errorbar(aux[1][i], aux2[0][i][2], yerr = np.sqrt(aux2[1][i][2][2]), fmt = "ro")
        hh = np.arange(0,2,0.01)
        
        G1 = G*np.ones(len(hh))
        nameG = "Calculated Residual Gas Damping at " + str("%0.1E" % G) + " [Pa]"
        plt.plot(hh, G1, "blue", label = nameG)
        plt.fill_between(hh, G1, facecolor='blue', alpha=0.5)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("dgx")
        plt.ylabel("$\Gamma$ [Hz]")
        plt.xlim(0.01,1)
        plt.legend(loc=3)
        plt.grid()
        plt.tight_layout(pad = 0)

plot_all(path_list)
plt.show()
