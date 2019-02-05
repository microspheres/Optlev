import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt

path_list = [r"C:\data\20190202\15um\4\PID\comx1"]


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

	return [x, trigger, PID]

    
def get_files_path(path):
    file_list = glob.glob(path+"\*.h5")
    return file_list

def get_data_path(path):
    A = []
    for i in get_files_path(path):
        a = getdata(i)
        A.append(a)
    return A

def trigger_on(path): # return index of PID ON
    F = get_data_path(path)
    index = []
    for i in range(len(F)):
        indx = np.where(np.diff(F[i][1]) > 2)
        plus = np.ones(len(indx))
        indx = plus + indx
        index.append(indx)
    return index

def trigger_off(path): # return index of PID off
    F = get_data_path(path)
    index = []
    for i in range(len(F)):
        indx = np.where(np.diff(F[i][1]) < -2)
        plus = np.ones(len(indx))
        indx = plus + indx
        index.append(indx)
    return index

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
        oo = trigger_on(path)[j][0]
        ff = trigger_off(path)[j][0]
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
        oo = trigger_on(path)[j][0]
        ff = trigger_off(path)[j][0]
        x = get_data_path(path)[j]
        for i in range(len(oo)-1): # a plot for every trigger on
            y = D[j][0][int(oo[i]):int(ff[i+1])]
            t = range(len(y))
            Y.append(y)
            X.append(t)
        Xf.append(X)
        Yf.append(Y)
    return [Xf,Yf]


plt.figure()
plt.plot(get_data_path(path_list[0])[0][1])
plt.plot(np.diff(get_data_path(path_list[0])[0][1]))
plt.plot(get_data_path(path_list[0])[0][0])

a = plot_PID_on(path_list[0])

plt.figure()
for i in range(len(a[0][0])):
    plt.plot(a[0][0][i], a[1][0][i])
plt.show()

