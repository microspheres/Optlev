
## load all files in a directory and plot the correlation of the resonse
## with the drive signal versus time

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, time, glob
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle

from scipy.optimize import curve_fit

path = r"C:\data\20170511\bead2_15um_QWP\new_sensor_feedback\charge31_freqcomb_piezo_150.0_74.9_75.4"
ts = 1.

fdrive = 41.
make_plot = True

data_columns = [0, 0] ## column to calculate the correlation against
drive_column = 5 ##-1 ## column containing drive signal

def getphase(fname):
        print "Getting phase from: ", fname 
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))
        fsamp = attribs["Fsamp"]
        xdat = dat[:,data_columns[1]]

        xdat = np.append(xdat, np.zeros( int(fsamp/fdrive) ))
        driver = dat[:,drive_column]
        Ddriver = np.gradient(driver)
        corr2 = np.correlate(xdat,driver*Ddriver)
        maxv = np.argmax(corr2) 

        cf.close()

        print maxv
        return maxv


def getdata(fname, maxv):

	print "Processing ", fname
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))

        if( len(attribs) > 0 ):
            fsamp = attribs["Fsamp"]

        xdat = dat[:,data_columns[1]]

        lentrace = len(xdat)
        ## zero pad one cycle
        MinusDC = dat[:,drive_column]
        corr_full = bu.corr_func(MinusDC - np.median(MinusDC), xdat, fsamp, fdrive)

        #plt.figure()
        #plt.plot( xdat)
        #plt.plot(dat[:,drive_column])
        #plt.show()
        

        return corr_full[0], np.max(corr_full) 

def get_most_recent_file(p):

    ## only consider single frequency files, not chirps
    filelist = glob.glob(os.path.join(p,"*.h5"))  ##os.listdir(p)
    #filelist = [filelist[0]]
    mtime = 0
    mrf = ""
    for fin in filelist:
        if( fin[-3:] != ".h5" ):
            continue
        f = os.path.join(path, fin) 
        if os.path.getmtime(f)>mtime:
            mrf = f
            mtime = os.path.getmtime(f)

    fnum = re.findall('\d+.h5', mrf)[0][:-3]
    return mrf#.replace(fnum, str(int(fnum)-1))
    

    
def list_file_time_order(p):
    #filelist = glob.glob(os.path.join(p,"*20Vpp*.h5"))
    filelist = glob.glob(os.path.join(p,"*.h5"))
    filelist.sort(key=os.path.getmtime)
    return filelist
    
 
best_phase = []
corr = []
   
for i in np.arange(len(list_file_time_order(path))):
        best_phase.append(getphase( list_file_time_order(path)[i] ))
        corr.append(getdata(list_file_time_order(path)[i],best_phase[i]))

#time = []
#T = 0
#
#for i in np.arange(len(list_file_time_order(path)) - 1):
#    t = os.path.getmtime(list_file_time_order(path)[i+1])-os.path.getmtime(list_file_time_order(path)[i])
#    T = T + t
#    time.append(T)

def Line(A, B):
    return 0.0*A + 1.0*B

X, Y = zip(*corr)

print 'mean'
print np.mean(X)
print 'std'
print np.std(X)/np.sqrt(len(X))


plt.plot(corr)
plt.grid()
plt.show()


# time = []
# for i in np.arange(len(list_file_time_order(path))):
#     i = i+1
#     time.append(i)

# coor2 = []
# for i in np.arange(len(list_file_time_order(path))):
#     a = corr[i][0]
#     coor2.append(a)
    

# popt1, pcov1 = curve_fit(Line, time[31:42], coor2[31:42])

# popt2, pcov2 = curve_fit(Line, time[48:58], coor2[48:58])

# popt3, pcov3 = curve_fit(Line, time[63:72], coor2[63:72])
