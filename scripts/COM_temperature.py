import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt
# import Volt_to_meter_single_e as vm

# path300k = r"C:\data\20190202\15um\4"
# name300k = r"2mbar_yzcool.h5"

# path_save = r"C:\data\20190202\15um\4\PID"

charge = True
if charge:
        path_1e = r"C:\data\20190304\15um_low532\6\1electron"
        path_ee = r"C:\data\20191014\22um\prechamber_LP\5\calibration1e"

path300k = r"C:\data\20190408\15um\3"
name300k = r"2mbar_yzcool.h5"

path300k = r"C:\data\20191014\22um\prechamber_LP\5"
name300k = r"2mbar_zcool.h5"

path_save = r"C:\data\20191014\22um\prechamber_LP\5\Temperature_X"

no_sphere = False
if no_sphere:
        pathno = [r"C:\data\20190326\15um_low532_50x\3\temp\no_sphere"]
        fileno = r"nosphere.h5"

f_start = 60. # for the fit
f_end = 140. # for the fit

NFFT = 2**16

kb = 1.38*10**-23

mass = 2.*2.3*10**-26

vis = 2.98e-5
vis_hidrogen = 1.37e-5

rho = 1800.0

R = 7.5*10**-6

M = (4./3.)*np.pi*(R**3)*rho

press = 150.

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

# fit_points1 = np.logical_and(data[0] > f_start, data[0] < 59.6)
# fit_points2 = np.logical_and(data[0] > 60.6 , data[0] < 65.0)
# fit_points3 = np.logical_and(data[0] > 65.7 , data[0] < 95.5)
# fit_points4 = np.logical_and(data[0] > 96.2 , data[0] < 119.0)

# fit_points5 = np.logical_and(data[0] > 121 , data[0] < f_end)
# fit_points_new = fit_points1+fit_points2+fit_points3+fit_points4

fit_points1 = np.logical_and(data[0] > f_start, data[0] < 59.0)
fit_points2 = np.logical_and(data[0] > 61.0, data[0] < 119.0)
fit_points3 = np.logical_and(data[0] > 121.0, data[0] < 122.0)
fit_points4 = np.logical_and(data[0] > 123.3, data[0] < 144.8)
fit_points5 = np.logical_and(data[0] > 145,9, data[0] < 179.0)
fit_points6 = np.logical_and(data[0] > 181.0, data[0] < f_end)
fit_points_new = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5

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
    C = (1. + 0.31*Kn(vis, press, temp, mass, R)/(0.785 + 1.152*Kn(vis, press, temp, mass, R) + Kn(vis, press, temp, mass, R)**2) )
    return A*B*C

def psd(f, A, f0, gammaover2pi):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    gamma = 2.*np.pi*gammaover2pi
    s1 = 2.*kb*temp*(gamma*(w0**2))
    s2 = 1.*M*(w0**2)*((w0**2 - w**2)**2 + (gamma*w)**2)
    s = np.sqrt(s1/s2)
    return A*s

gamma = Gamma(vis, press, temp, mass, R, M)

px, cx = opt.curve_fit(psd, data[0][fit_points_new], np.sqrt(data[1][fit_points_new]), p0 = [1e6, 80., gamma] )

f = np.arange(f_start, f_end, 1e-2)

############################ folder with temperatures
# path_list = [r"C:\data\20190202\15um\4\PID\full1", r"C:\data\20190202\15um\4\PID\full2", r"C:\data\20190202\15um\4\PID\full3",r"C:\data\20190202\15um\4\PID\full4", r"C:\data\20190202\15um\4\PID\full5", r"C:\data\20190202\15um\4\PID\full6", r"C:\data\20190202\15um\4\PID\full7", r"C:\data\20190202\15um\4\PID\full8", r"C:\data\20190202\15um\4\PID\full9", r"C:\data\20190202\15um\4\PID\full10", r"C:\data\20190202\15um\4\PID\full11", r"C:\data\20190202\15um\4\PID\full12"]

# path_list = [r"C:\data\20190211\15um\1\lp\1", r"C:\data\20190211\15um\1\lp\2", r"C:\data\20190211\15um\1\lp\3", r"C:\data\20190211\15um\1\lp\4", r"C:\data\20190211\15um\1\lp\5", r"C:\data\20190211\15um\1\lp\6", r"C:\data\20190211\15um\1\lp\7", r"C:\data\20190211\15um\1\lp\8", r"C:\data\20190211\15um\1\lp\9", r"C:\data\20190211\15um\1\lp\10", r"C:\data\20190211\15um\1\lp\11"]

# path_list = [r"C:\data\20190326\15um_low532_50x\3\temp\1", r"C:\data\20190326\15um_low532_50x\3\temp\2", r"C:\data\20190326\15um_low532_50x\3\temp\3", r"C:\data\20190326\15um_low532_50x\3\temp\4",r"C:\data\20190326\15um_low532_50x\3\temp\5", r"C:\data\20190326\15um_low532_50x\3\temp\6", r"C:\data\20190326\15um_low532_50x\3\temp\7", r"C:\data\20190326\15um_low532_50x\3\temp\8", r"C:\data\20190326\15um_low532_50x\3\temp\9", r"C:\data\20190326\15um_low532_50x\3\temp\10", r"C:\data\20190326\15um_low532_50x\3\temp\11trekoff", r"C:\data\20190326\15um_low532_50x\3\temp\12", r"C:\data\20190326\15um_low532_50x\3\temp\13", r"C:\data\20190326\15um_low532_50x\3\temp\14", r"C:\data\20190326\15um_low532_50x\3\temp\15",r"C:\data\20190326\15um_low532_50x\3\temp\16" ]

path_list = [r"C:\data\20190408\15um\3\temp\1", r"C:\data\20190408\15um\3\temp\2", r"C:\data\20190408\15um\3\temp\3", r"C:\data\20190408\15um\3\temp\4", r"C:\data\20190408\15um\3\temp\5", r"C:\data\20190408\15um\3\temp\6", r"C:\data\20190408\15um\3\temp\7", r"C:\data\20190408\15um\3\temp\8", r"C:\data\20190408\15um\3\temp\9", r"C:\data\20190408\15um\3\temp\10", r"C:\data\20190408\15um\3\temp\11", r"C:\data\20190408\15um\3\temp\12", r"C:\data\20190408\15um\3\temp\13", r"C:\data\20190408\15um\3\temp\14"]

path_list = [r"C:\data\20191014\22um\prechamber_LP\5\Temperature_X\1", r"C:\data\20191014\22um\prechamber_LP\5\Temperature_X\2", r"C:\data\20191014\22um\prechamber_LP\5\Temperature_X\3",r"C:\data\20191014\22um\prechamber_LP\5\Temperature_X\4",r"C:\data\20191014\22um\prechamber_LP\5\Temperature_X\5",r"C:\data\20191014\22um\prechamber_LP\5\Temperature_X\6", r"C:\data\20191014\22um\prechamber_LP\5\Temperature_X\6",r"C:\data\20191014\22um\prechamber_LP\5\Temperature_X\8", ]


######### Gamma low pressure
PP = getdata(glob.glob(path_list[0]+"\*.h5")[0])[3]
G = Gamma(vis_hidrogen, 100.*PP, temp, mass, R, M)
Gover2pi = G/(2*np.pi)

####################


def get_files_path(path):
        file_list = glob.glob(path+"\*.h5")
        return file_list


def get_data_path(path):
        info = getdata(get_files_path(path)[0])
        freq = info[0]
        dgx = info[2][0]
        Xpsd = np.zeros(len(freq))
        aux = get_files_path(path)
        for i in aux:
                a = getdata(i)
                Xpsd += a[1]
                p = a[3]
        Xpsd =  Xpsd/len(aux)
        return [Xpsd, dgx, p]
        

def psd_paths(pathlist):
        dgx = []
        xpsd = []
        P = []
        for i in pathlist:
               a = get_data_path(i)
               dgx.append(a[1])
               xpsd.append(a[0])
               P.append(a[2])
        return [xpsd, dgx, P]

def fit_paths(pathlist):
        A = psd_paths(pathlist)[0]
        PX = []
        CX = []
        for i in range(len(A)):
                print px[1]
                px_aux, cx_aux = opt.curve_fit(psd, data[0][fit_points_new], np.sqrt(A[i][fit_points_new]), p0 = [1e6, abs(px[1]), 100.], bounds = ((0.1, abs(px[1])-10., 0.0001),(1e6, abs(px[1]+30), 100000)) )
                CX.append(cx_aux)
                PX.append(px_aux)
        return [PX, CX]

def plot_all(pathlist):
        plt.figure()
        name_hp = "dgx = " + str("%.1E" % data[2][0]) + " $\Gamma$ = " + str("%.1E" % px[2]) + " Hz" + " $f_0$ = " + str("%.1E" % abs(px[1])) + " $\pm$ " + str("%.0E" % np.sqrt(abs(cx[1][1])))  +  " Hz"
        plt.loglog(data[0], np.sqrt(data[1])/px[0], label = name_hp)
        print "conversion[V/m] = ", px[0]
        plt.loglog(f, psd(f,*px)/px[0])
        aux = psd_paths(pathlist)
        aux2 = fit_paths(pathlist)
        for i in range(len(pathlist)):
                if i == i:
                        name = "dgx = " + str("%.1E" % aux[1][i]) + " $\Gamma$ = " + str("%.2E" % aux2[0][i][2]) + " Hz" + " $f_0$ = " + str("%.2E" % aux2[0][i][1]) + " $\pm$ " + str("%.0E" % np.sqrt(aux2[1][i][1][1])) + " Hz"
                        plt.loglog(data[0], np.sqrt(aux[0][i])/px[0], label = name)
                        plt.loglog(f, psd(f, *aux2[0][i])/px[0])

        plt.xlim(1,500)
        plt.ylim(2e-13,2e-8)

        if no_sphere:
                No = getdata(os.path.join(pathno[0], fileno))
                plt.loglog(No[0], np.sqrt(No[1])/px[0], label = "No sphere")
                pno, cno = opt.curve_fit(psd, No[0][fit_points_new], np.sqrt(No[1][fit_points_new]), p0 = [1e6, abs(px[1]), gamma], bounds = ([1e2, abs(px[1])-15., 0.0001*gamma], [1e10, abs(px[1])+15., 500.*gamma]) )
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
                plt.loglog(aa, 1e6*tt, "b", alpha = 0.5, label = "Imaging System Noise")
                plt.fill_between(aa, 1e6*tt, facecolor='blue', alpha=0.5)
        plt.xlabel("dgx")
        plt.ylabel("COMx Temp [$\mu$K]")
        plt.xlim(0.001,2)
        plt.legend(loc=3)
        plt.grid()
        plt.tight_layout(pad = 0)
        return [dg, T]

        ########

        plt.figure()
        for i in range(len(pathlist)):
                plt.errorbar(aux[1][i], aux2[0][i][2], yerr = np.sqrt(aux2[1][i][2][2]), fmt = "ro")
        hh = np.arange(0,2,0.01)
        
        G1 = Gover2pi*np.ones(len(hh))
        nameG = "Calculated Residual Gas Damping at " + str("%0.1E" % PP) + " [mbar]"
        plt.plot(hh, G1, "blue", label = nameG)
        # plt.fill_between(hh, G1, facecolor='blue', alpha=0.5)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("dgx")
        plt.ylabel("$\Gamma$ [Hz]")
        plt.xlim(0.01,1)
        plt.legend(loc=3)
        plt.grid()
        plt.tight_layout(pad = 0)

        
        name = str(path_save) + "\Gamma_from_com_temp.npy"
        Dg = []
        Dam = []
        Daerr = []
        for i in range(len(pathlist)):
                dgx = aux[1][i]
                dm = aux2[0][i][2]
                derr = np.sqrt(aux2[1][i][2][2])
                Dg.append(dgx)
                Dam.append(dm)
                Daerr.append(derr)
        a = np.array([Dg, Dam, Daerr, Gover2pi, PP])
        np.save(name , a)



def plot_COM_temp_high_pressure(pathlist):
        ## get P and xpsd
        P = []
        X = []
        F = []
        for i in pathlist:
                flist = get_files_path(i)
                xpsd = 0
                for j in flist:
                        A = getdata(j)
                        p = A[3]
                        freq = A[0]
                        xpsd += A[1]
                F.append(freq)
                X.append(xpsd)
                P.append(p)
        fit_points_new2 = np.logical_and(data[0] > 30., data[0] < 300.)
        i = 0
        popt, pcov = opt.curve_fit(psd, F[0][fit_points_new2], np.sqrt(X[i][fit_points_new2]))
        g = popt[2]
        plt.figure()
        gth = Gamma(vis, 100.*P[i], temp, mass, R, M)/(2.*np.pi)
        name = "press = " + str(P[i]) + " mbar"
        name2 = "Gamma/2pi = " + str(g) + " Hz"
        plt.loglog(F[0], np.sqrt(X[i]), label = name)
        plt.loglog(F[0][fit_points_new2], psd(F[0][fit_points_new2], *popt), label = name2)
        plt.legend()
        print gth
        
        
        # plt.figure()
        # for i in range(len(pathlist)):
        #         plt.loglog(F[i], np.sqrt(X[i]))

        # plt.figure()
        # T = []
        # PP = []
        # for i in range(len(pathlist)):
        #         T.append(np.sum(X[i]))
        #         PP.append(P[i])
        # plt.plot(PP, T, "ro")
        
                

#plot_COM_temp_high_pressure(path_list)
info = plot_all(path_list)

# if charge:
#         acc = vm.acc(path_1e)
#         aux = vm.findAC_peak(path_1e)
#         peakpos = aux[0]
#         freq_0 = aux[1]
#         aux2 = vm.get_data_path(path_1e)
#         freq = aux2[4]
#         xpsd_volt = aux2[0]
#         x_amp_volt = np.sum(xpsd_volt[peakpos - 3:peakpos +3])
#         x_amp_volt = np.sqrt(x_amp_volt) # not dividing by pi because I am comparing to the conversion made above.

#         print x_amp_volt

#         Z = np.sqrt( (630.)**2 + (1/((2.*np.pi*freq_0)**2))*((2.*np.pi*px[1])**2 - (2.*np.pi*freq_0)**2)**2  )

#         x_m = acc/(Z*(2.*np.pi*freq_0))

#         convertion_v_to_m = x_amp_volt/x_m

#         print convertion_v_to_m

#         plt.figure()
#         plt.loglog(data[0], np.sqrt(data[1])/px[0], label = "psd conversion")
#         plt.loglog(data[0], np.sqrt(data[1])/convertion_v_to_m, label = "1e conversion")
#         plt.legend()

        
        



        
plt.show()
