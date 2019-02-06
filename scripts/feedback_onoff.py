import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt

path_list = [r"C:\data\20190202\15um\4\PID\COMx1", r"C:\data\20190202\15um\4\PID\COMx2", r"C:\data\20190202\15um\4\PID\COMx3", r"C:\data\20190202\15um\4\PID\COMx4", r"C:\data\20190202\15um\4\PID\COMx5", r"C:\data\20190202\15um\4\PID\COMx6", r"C:\data\20190202\15um\4\PID\COMx7", r"C:\data\20190202\15um\4\PID\COMx8", r"C:\data\20190202\15um\4\PID\COMx9", r"C:\data\20190202\15um\4\PID\COMx10", r"C:\data\20190202\15um\4\PID\COMx11"]

plot = False
plot_heat = False
plot_cool = False


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

	x = dat[:, bu.xi]-numpy.mean(dat[:, bu.xi])
        trigger = dat[:, 4]

	return [x, trigger, PID, Fs]

Q = getdata(glob.glob((path_list[0]+"\*.h5"))[0])
fs = Q[3]

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
            t = range(len(y))
            Y.append(y)
            X.append(t)
        Xf.append(X)
        Yf.append(Y)
    return [Xf,Yf]

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
            t = range(len(y))
            Y.append(y)
            X.append(t)
        Xf.append(X)
        Yf.append(Y)
    return [Xf,Yf]

def fit_cool(time, A, X, w0, phase):
    a = 1.0*A*(np.exp(-X*w0*time))*np.sin( np.sqrt(1 - X**2)*w0*time + phase)
    return a

def fit_heat(time, A, X, w0, phase):
    a = 1.0*A*(np.exp(X*w0*time))*np.sin( np.sqrt(1 - X**2)*w0*time + phase)
    return a

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
            p1 = np.array([0.1, 0.3, 2*np.pi*80, np.pi/2])
            notfail = True
            try:
                p, c = opt.curve_fit(fit_cool, np.array(a[0][j][i])/fs, a[1][j][i], p0 = p1, bounds = ((-1,0.00002,2*np.pi*60, 0),(1, 0.99, 2*np.pi*95, np.pi)))
            except RuntimeError:
                c = 0
                p = p1
                print "FIT FAIL"
                notfail = False
            if plot:
                plt.figure()
                plt.plot(np.array(a[0][0][i])/fs, a[1][0][i])
                plt.plot(np.array(a[0][0][i])/fs, fit_cool(np.array(a[0][0][i]/fs), *p), "k--")
            if notfail:
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
            p1 = np.array([0.01, 0.03, 2*np.pi*70, 0.])
            notfail = True
            try:
                p, c = opt.curve_fit(fit_heat, np.array(a[0][j][i])/fs, a[1][j][i], p0 = p1)
            except RuntimeError:
                c = 0
                p = p1
                notfail = False
                print "FIT FAIL"
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
        label1 = "dgx = " + str("%0.2f" % dgx) + "mean damping = " + str( "%0.1f" % np.mean(Info[i][1][0]) )
        label2 = "OFFdgx = " + str("%0.2f" % dgx) + "mean heating = " + str( "%0.1f" % np.mean(Info[i][0][0]) )
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

plot_all(path_list)

# A = get_from_pathlist(path_list)
# plt.figure()
# plt.plot(A[0][0][0])
# plt.figure()
# plt.plot(A[0][1][0])
# print A[0][2]
    
# damp = plot_and_fit_cool(path_list[0], plot)[2]
# plt.figure()
# plt.plot(damp)
# print "damp =", damp



# antidamp = plot_and_fit_heat(path_list[0], plot)[2][0]
# plt.figure()
# plt.plot(antidamp)
# print "antidamp =", np.mean(antidamp)


if plot:
    plt.legend(loc=3)
plt.show()
