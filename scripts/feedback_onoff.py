import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt
import multiprocessing as mp
from scipy.signal import butter, lfilter, filtfilt

################# THIS CODE REQUIRES THE TRIGGER AND FEEDBACK ON AND OFF


################# Failing fits are NOT included in the final result =D happy face


############ input the corrent resonance freq limits at damping fit!!!

#path_list = [r"C:\data\20190202\15um\4\PID\COMx1"]

# path_list = [r"C:\data\20190202\15um\4\PID\COMx1", r"C:\data\20190202\15um\4\PID\COMx2", r"C:\data\20190202\15um\4\PID\COMx3", r"C:\data\20190202\15um\4\PID\COMx4", r"C:\data\20190202\15um\4\PID\COMx5", r"C:\data\20190202\15um\4\PID\COMx6", r"C:\data\20190202\15um\4\PID\COMx7", r"C:\data\20190202\15um\4\PID\COMx8", r"C:\data\20190202\15um\4\PID\COMx9", r"C:\data\20190202\15um\4\PID\COMx10", r"C:\data\20190202\15um\4\PID\COMx11"]

# path_list = [r"C:\data\20190326\15um_low532_50x\8\pid_onoff\X\1", r"C:\data\20190326\15um_low532_50x\8\pid_onoff\X\2", r"C:\data\20190326\15um_low532_50x\8\pid_onoff\X\3", r"C:\data\20190326\15um_low532_50x\8\pid_onoff\XY\1", r"C:\data\20190326\15um_low532_50x\8\pid_onoff\XY\2", r"C:\data\20190326\15um_low532_50x\8\pid_onoff\XY\3"]

#path_list = [r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\PID_ONOFF\2\X\1",r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\PID_ONOFF\2\X\2",r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\PID_ONOFF\2\X\3",r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\PID_ONOFF\2\X\4",r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\PID_ONOFF\2\X\5",r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\PID_ONOFF\2\X\6",r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\PID_ONOFF\2\X\7",r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\PID_ONOFF\2\X\8", r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\PID_ONOFF\2\X\9",r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\PID_ONOFF\2\X\10",r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\PID_ONOFF\2\X\11",]

# path_list = [r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\PID_ONOFF\2\X\1", r"C:\data\20190326\15um_low532_50x\10_2th_orderLPFF\PID_ONOFF\2\X\2",]

path_list = [r"C:\data\20190619\15um\3\feedback_onoff\1xy", ]

path_save = r"C:\data\20190619\15um\3\feedback_onoff\1xy"

# path_list = [r"C:\data\20190202\15um\4\PID\COMx10"]

plot = True
plot_heat = False
plot_cool = False
bins = 15
initial_threshould = 0.08 # helps the fit of the damping

f0 = 85. # for the fit, use the fit from the psd with low feedback
df0 = 3.0 # for the fit

gaussfit_heat = False # use a gaussian to fit the results of each folder (not always work because the shape is far from gaussian)
gaussfit_damp = False

LPfilter = False
order = 2
def butter_lowpass(lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a


def butter_lowpass_filter(data, lowcut, fs, order):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


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
		# Press = dset.attrs['pressures'][0]
                # print Press
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                PID = dset.attrs['PID']
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	x = dat[:, bu.xi]-numpy.mean(dat[:, bu.xi])
        if LPfilter:
                x = butter_lowpass_filter(x, 10.*f0, Fs, order)
        trigger = dat[:, 4]


	return [x, trigger, PID, Fs]

Q = getdata(glob.glob((path_list[0]+"\*.h5"))[1])
fs = Q[3]
# plt.figure()
# t = np.array(range(len(Q[0])))/fs
# plt.plot(t,0.1*Q[1], label = "Trigger")
# plt.plot(t,Q[0], label = "Signal X direction")
# plt.xlabel("Time [s]")
# plt.ylabel("Signal [V]")
# plt.legend(loc=3)
# plt.tight_layout(pad = 0)
# plt.show()

def get_files_path(path):
    file_list = glob.glob(path+"\*.h5")
    return file_list

def get_data_path(path):
    A = []
    for i in get_files_path(path):
        a = getdata(i)
        A.append(a)
    return A

# def trigger_on(path): # return index of PID ON: THIS IS FALTY
#     F = get_data_path(path)
#     index = []
#     for i in range(len(F)):
#         indx = np.where(np.diff(F[i][1]) > 2)
#         plus = np.ones(len(indx))
#         indx = plus + indx
#         index.append(indx)
#     return index

# def trigger_off(path): # return index of PID off: THIS IS FALTY
#     F = get_data_path(path)
#     index = []
#     for i in range(len(F)):
#         indx = np.where(np.diff(F[i][1]) < -2)
#         plus = np.ones(len(indx))
#         indx = plus + indx
#         index.append(indx)
#     return index




def trigger_on2(path): # return index of PID ON
    F = get_data_path(path)
    indexf = []
    for i in range(len(F)):
        index = []
        indx = np.where((F[i][1]) > 3.)
        for j in range(len(indx[0])-1):
            b = (indx[0][j+1] - indx[0][j])
            if b != 1:
                indx2 = float(indx[0][j+1])
                index.append(indx2)
        indexf.append(np.array(index))
    return indexf

def trigger_off2(path): # return index of PID Off
    F = get_data_path(path)
    indexf = []
    for i in range(len(F)):
        index = []
        indx = np.where((F[i][1]) < 3.)
        index.append(indx[0][0])  # this is necessary due to the fact that the file is saved with the first trigger ON
        for j in range(len(indx[0])-1):
            b = (indx[0][j+1] - indx[0][j])
            if b != 1:
                indx2 = float(indx[0][j+1])
                index.append(indx2)
        indexf.append(np.array(index))
    return indexf

def trigger(path):
    F = get_data_path(path)
    print len(F)
    index = []
    T = []
    for i in range(len(F)):
        a = trigger_on(path)[i][0]
        b = trigger_off(path)[i][0]
        t = np.sort(np.concatenate((a,b)))
        T.append(t)
    return T

def plot_PID_off(path): # return xxx[a][b][c] a is the X or Y axis, b is the file inside the folder and c is the next trigger
    X = []
    Y = []
    Xf = []
    Yf = []
    D = get_data_path(path)
    for j in range(len(D)): # for every file on the folder
        oo = trigger_on2(path)[j]
        ff = trigger_off2(path)[j]
        x = get_data_path(path)[j]
        for i in range(len(ff)): # a plot for every trigger off
            y = D[j][0][int(ff[i]):int(oo[i])]
            y = y -np.mean(y)
            t = range(len(y))
            Y.append(y)
            X.append(t)
        Xf.append(X)
        Yf.append(Y)
    return [Xf,Yf]

def accumulator_off(path_list): # the 2400 is to ensure that all trigger has the same lenght

    y = np.zeros(2400)
    y2 = y
    for i in path_list: # for folder in path list
        file_list = glob.glob(i+"\*.h5")
        a = plot_PID_off(i)
        t = a[0][0][0][0:2400]
        for j in range(len(file_list)): # for every h5 file inside the folder
            for k in range(len(a[0][j])): # for every trigger fb off
                y2 = (a[1][j][k][0:2400]**2  + y2)
                y = a[1][j][k][0:2400] + y

    def func_exp(t, A, B):
        return A*np.exp(t*B)

    t = t/fs

    popt, pcov = opt.curve_fit(func_exp, t, y2)
    print popt
    
    plt.figure()
    plt.semilogy(t, y2)
    plt.semilogy(t, func_exp(t, *popt))

    print "Gamma_heat[Hz] = ", popt[1]/2.


accumulator_off(path_list)
plt.show()

# a = plot_PID_off(path_list[0])
# plt.figure()
# print len(a[0][2])
# aaa = np.zeros(len(a[0][2][30][0:2400]))
# aaa2 = aaa
# for i in range(len(a[0][2])):
#     aaa2 = (a[1][2][i][0:2400]**2  + aaa2)
#     aaa = a[1][2][i][0:2400]


# aaa = aaa**2
# aaa = aaa2 - aaa
    
# plt.semilogy(a[0][2][30][0:2400], aaa)
# # plt.plot(a[0][2][30][0:2400], a[1][2][30][0:2400])
# plt.show()

def plot_PID_on(path): # return xxx[a][b][c] a is the X or Y axis, b is the file inside the folder and c is the next trigger
    X = []
    Y = []
    Xf = []
    Yf = []
    D = get_data_path(path)
    for j in range(len(D)): # for every file on the folder
        oo = trigger_on2(path)[j]
        ff = trigger_off2(path)[j]
        x = get_data_path(path)[j]
        for i in range(len(oo)-1): # a plot for every trigger on
            y = D[j][0][int(oo[i]):int(ff[i+1])]
            y = y - np.mean(y)
            t = range(len(y))
            Y.append(y)
            X.append(t)
        Xf.append(X)
        Yf.append(Y)
    return [Xf,Yf]

def fit_cool(time, A, X, w0, phase, A2, c): # has to consider more than damping case.
        if X >= 1:
                a = 1.0*A*np.exp((-X*w0 - w0*np.sqrt(X**2 - 1.))*time) + 1.0*A2*np.exp((-X*w0 + w0*np.sqrt(X**2 - 1.))*time) + c
        else:
                a = 1.0*A*(np.exp(-X*w0*time))*(np.sin( np.sqrt(1 - X**2)*w0*time + phase)) + c
        return a


def fit_heat(time, A, X, w0, phase, c):
    a = 1.0*A*(np.exp(X*w0*time))*np.sin( np.sqrt(1 - X**2)*w0*time + phase) + c
    return a

def residuals(a,b):
        R = a - b
        r = 0
        for i in R:
                r = r + i**2
        return r

def plot_and_fit_cool(path, plot):
    Pf = []
    Cf = []
    C = []
    P = []
    a = plot_PID_on(path)
    Df = []
    for j in range(len(a[0])):
        D = []
        for i in range(len(a[0][j])):
            c1 = 0
            p1 = np.array([0.1, 1.2, 2*np.pi*f0, 0.1, 0.1, 0])
            p1a = np.array([0.1, 1.7, 2*np.pi*f0, 0.1, 0.1, 0])
            p1b = np.array([0.1, 0.01, 2*np.pi*f0, 0.1, 0.1, 0])
            notfail = True
            aux_a = False
            aux_b = False
            if np.abs(np.mean(a[1][j][i][0:10])) < initial_threshould:
                    notfail = False
                    print "initial fail"
            else:
                try:
                        p_a, c_a = opt.curve_fit(fit_cool, np.array(a[0][j][i])/fs, a[1][j][i], p0 = p1a, bounds = ((-2, 1.0, 2*np.pi*(f0 - 5*df0), 0, -2, -0.5),(2, 6., 2*np.pi*(f0 + 5*df0), np.pi, 2, 0.5)))
                        ra = residuals(a[1][j][i], fit_cool(np.array(a[0][j][i])/fs , *p_a))
                except:
                        ra = 1e13
                        print "FAIL COOL RA"
                        aux_a = True
                try:
                        p_b, c_b = opt.curve_fit(fit_cool, np.array(a[0][j][i])/fs, a[1][j][i], p0 = p1b, bounds = ((-2, 0.0002, 2*np.pi*(f0 - 5*df0), 0, -2, -0.5),(2, 0.99, 2*np.pi*(f0 + 5*df0), np.pi, 2, 0.5)))
                        rb = residuals(a[1][j][i], fit_cool(np.array(a[0][j][i])/fs , *p_b))
                except:
                        rb = 1e13
                        print "FAIL COOL RB"
                        aux_b = True

                if aux_a and aux_b:
                        notfail = False
                        print "double_fail_cool"
                if ra >= rb:
                        p = p_b
                        c = c_b
                else:
                        p = p_a
                        c = c_a

            if notfail:
                if plot:
                        plt.figure()
                        label = "gamma(rad/s) =" + str(2.*p[1]*p[2])
                        plt.plot(np.array(a[0][0][i])/fs, a[1][0][i])
                        plt.plot(np.array(a[0][0][i])/fs, fit_cool(np.array(a[0][0][i]/fs), *p), "k--", label = label)
                        plt.legend()
                P.append(p)
                C.append(c)
                damp = 2.*p[1]*p[2]
                D.append(damp)
        Df.append(D)
        Pf.append(P)
        Cf.append(C)
    return [Pf, Cf, Df]

def plot_and_fit_heat(path, plot):
    Pf = []
    Cf = []
    C = []
    P = []
    a = plot_PID_off(path)
    Df = []
    for j in range(len(a[0])):
        D = []
        for i in range(len(a[0][j])):
            c1 = 0
            p1 = np.array([0.03, 0.02, 2*np.pi*f0, 0.01, 0.])
            notfail = True
            try:
                p, c = opt.curve_fit(fit_heat, np.array(a[0][j][i])/fs, a[1][j][i], p0 = p1)
            except RuntimeError:
                c = 0
                p = p1
                notfail = False
                print "FIT FAIL HEAT"
            if plot:
                plt.figure()
                plt.plot(np.array(a[0][0][i])/fs, a[1][0][i])
                plt.plot(np.array(a[0][0][i])/fs, fit_heat(np.array(a[0][0][i]/fs), *p), "k--")
            if notfail:
                P.append(p)
                C.append(c)
                damp = 2.*p[1]*p[2]
                D.append(damp)
        Pf.append(P)
        Cf.append(C)
        Df.append(D)
    return [Pf, Cf, Df]


def get_from_pathlist(pathlist): ### return [a][b] a is the folder and b is 0 for heating rate, 1 for damping rate and 2 for dgx of that folder.
    Info = []
    L = len(pathlist)
    for i in range(L):
        heat = plot_and_fit_heat(path_list[i], plot_heat)[2]
        damp = plot_and_fit_cool(path_list[i], plot_cool)[2]
        dgx_aux = getdata(glob.glob((pathlist[i]+"\*.h5"))[0])[2][0]
        info = np.array([heat, damp, dgx_aux])
        Info.append(info)
    return Info

def plot_all(pathlist):
    Info = get_from_pathlist(pathlist)
    L = len(pathlist)
    for i in range(L):
        dgx = Info[i][2]
        label1 = "dgx = " + str("%0.2f" % dgx) + " mean damping = " + str( "%0.1f" % np.mean(Info[i][1][0]) ) + " [Hz]"
        label2 = "OFFdgx = " + str("%0.2f" % dgx) + " mean heating = " + str( "%0.1f" % np.mean(Info[i][0][0]) ) + " [Hz]"
        plt.figure()
        plt.plot(Info[i][1][0], "ro", label = label1)
        plt.xlabel("Measurements")
        plt.ylabel("Damping [Hz]")
        plt.legend(loc=3)
        plt.grid()
        plt.tight_layout(pad = 0)
        plt.figure()
        plt.plot(Info[i][0][0], "ro", label = label2)
        plt.xlabel("Measurements")
        plt.ylabel("Heating [Hz]")
        plt.legend(loc=3)
        plt.grid()
        plt.tight_layout(pad = 0)
    return Info

def gauss(x,a,b,c):
    g = c*np.exp(-0.5*((x-a)/b)**2)
    return g

def poisson(x,l,c):
        from scipy.special import factorial
        g = c*((x**l)*np.exp(-l))/(factorial(x))
        return g

def plot_all_histogram(pathlist, plot):
    Info = get_from_pathlist(pathlist)
    L = len(pathlist)
    Dx = []
    dmean = []
    derror = []
    hmean = []
    herror = []
    list_damp_fail = []
    for i in range(L):
        dgx = Info[i][2]
        damp = Info[i][1][0]
        heat = Info[i][0][0]

        if gaussfit_damp:
                try:
                        #hist for damping
                        h,b = np.histogram(damp, bins = bins)
                        bc = np.diff(b)/2 + b[:-1]
                        p0 = np.array([np.mean(damp), np.std(damp)/np.sqrt(len(damp)), 10])
                        poptd, pcovd = opt.curve_fit(gauss, bc, h, p0 = p0)


                        mean_damp = np.mean(damp)
                        error_damp = np.std(damp)/np.sqrt(len(damp))
                        print "gauss_fit_damp_failed", pathlist[i]
                        
                        if np.abs((poptd[0] - np.mean(damp))) < 3.*poptd[1]:
                                error_damp = np.sqrt(pcovd[0][0])
                                mean_damp = poptd[0]
                                
                
                                label1 = "dgx = " + str("%0.2f" % dgx) + " damping = " + str( "%0.1f" % poptd[0] ) + "$\pm$" + str( "%0.1f" % np.sqrt(pcovd[0][0]) ) + " [Hz]"
                                space = np.arange(np.mean(damp) - 1000, np.mean(damp) + 1000, 0.1)
                                if plot:
                                        plt.figure()
                                        plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = 'ko')
                                        plt.plot(space, gauss(space,*poptd))
                                        plt.xlabel("Histogram Damping [Hz]")
                                        plt.ylabel("Measurements")
                                        plt.legend(loc=3)
                                        plt.grid()
                                        plt.tight_layout(pad = 0)
                except:
                        mean_damp = np.mean(damp)
                        error_damp = np.std(damp)/np.sqrt(len(damp))
                        print "gauss_fit_damp_failed", pathlist[i]
        else:
                mean_damp = np.mean(damp)
                error_damp = np.std(damp)/np.sqrt(len(damp))
                        
        if gaussfit_heat:
                try:
                        #hist for heating
                        h,b = np.histogram(heat, bins = bins)
                        bc = np.diff(b)/2 + b[:-1]
                        p0 = np.array([np.mean(heat), np.std(heat)/np.sqrt(len(heat)), 10])
                
                        popth, pcovh = opt.curve_fit(gauss, bc, h, p0 = p0)

                        mean_heat = np.mean(heat)
                        error_heat = np.std(heat)/np.sqrt(len(heat))
                        print "gauss_fit_heat_failed"

                        if np.abs((popth[0] - np.mean(heat))) < 3.*popth[1]:
                                mean_heat = popth[0]
                                error_heat = np.sqrt(pcovh[0][0])
                                
                
                                label1 = "dgx = " + str("%0.2f" % dgx) + " heating = " + str( "%0.1f" % popth[0] ) + "$\pm$" + str( "%0.1f" % np.sqrt(pcovh[0][0]) ) + " [Hz]"
                                space = np.arange(np.mean(heat) - 1000, np.mean(heat) + 1000, 0.1)
                                if plot:
                                        plt.figure()
                                        plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = 'ko')
                                        plt.plot(space, gauss(space,*popth))
                                        plt.xlabel("Histogram Heating [Hz]")
                                        plt.ylabel("Measurements")
                                        plt.legend(loc=3)
                                        plt.grid()
                                        plt.tight_layout(pad = 0)
                except:
                        mean_heat = np.mean(heat)
                        error_heat = np.std(heat)/np.sqrt(len(heat))
                        print "gauss_fit_heat_failed"
        else:
                mean_heat = np.mean(heat)
                error_heat = np.std(heat)/np.sqrt(len(heat))

                
        Dx.append(dgx)
        dmean.append(mean_damp)
        derror.append(error_damp)
        hmean.append(mean_heat)
        herror.append(error_heat)

    return [Dx, dmean, hmean, derror, herror]


def final_plot(pathlist, plot):
    para = plot_all_histogram(pathlist, plot)
    plt.figure()

    hm =  np.array(para[2])/(2.*np.pi)
    her = np.array(para[4])/(2.*np.pi)

    plt.errorbar(para[0], hm, yerr = her, fmt = "bo", label = "Heating")

    # careful for the damping: the measurement gives an effective damping. The real one has to consider the heating.

    dm = np.array(para[2])/(2.*np.pi) + np.array(para[1])/(2.*np.pi)
    der =  np.sqrt((np.array(para[3])/(2.*np.pi))**2 + (np.array(para[4])/(2.*np.pi))**2)
    
    plt.errorbar(para[0], dm, yerr = der, fmt = "ro", label = "Damping")
    plt.xlabel("Derivative gain X axis")
    plt.ylabel("$\Gamma / 2\pi$ [Hz]")
    plt.legend()
    plt.ylim(-10, 950)
    plt.grid()
    plt.tight_layout(pad = 0)


    a = np.array([para[0], hm, her, dm, der])
    name = str(path_save) + "\output_from_feed_backonoff.npy"
    np.save(name , a)
    
    return a

# def final_plot_multiprocessing(pathlist):

#     para = plot_all_histogram(pathlist, plot)


#     hm =  np.array(para[2])/(2.*np.pi)
#     her = np.array(para[4])/(2.*np.pi)

#     # careful for the damping: the measurement gives an effective damping. The real one has to consider the heating.

#     dm = np.array(para[2])/(2.*np.pi) + np.array(para[1])/(2.*np.pi)
#     der =  np.sqrt((np.array(para[3])/(2.*np.pi))**2 + (np.array(para[4])/(2.*np.pi))**2)

#     a = np.array([para[0], hm, her, dm, der])
#     name = str(path_save) + "\output_from_feed_backonoff.npy"
#     np.save(name , a)
    
#     return a
    

# def creat_list_multiprocessing(pathlist, plot):
#         A = []
#         for i in pathlist:
#                 a = [i, plot]
#                 A.append(a)
#         return A



# def main():
#         # AA = creat_list_multiprocessing(path_list, plot)
#         # print AA
#         pool = mp.Pool(processes = 1)
#         # print path_list
#         pool.map(final_plot_multiprocessing, path_list)


# if __name__ == "__main__":
#         mp.freeze_support()
#         main()
        
# if True:
#         plt.legend(loc=3)
# plt.show()




final_plot(path_list, plot)
if plot:
        plt.legend(loc=3)
plt.show()
