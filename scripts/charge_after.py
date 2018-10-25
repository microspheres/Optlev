
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

path = r"C:\data\20180702\bead2_SiO2_15um_POL_NS\temp_charge"
ts = 1.

fdrive = 48. #31.
make_plot = True

data_columns = [0, bu.xi] # column to calculate the correlation against
drive_column = bu.drive # column containing drive signal

def getphase(fname):
        print "Getting phase from: ", fname 
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))
        fsamp = attribs["Fsamp"]
        xdat = dat[:,data_columns[1]]

        xdat = np.append(xdat, np.zeros( int(fsamp/fdrive) ))
        corr2 = np.correlate(xdat,dat[:,drive_column])
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

filelist = glob.glob(os.path.join(path,"*.h5"))  ##os.listdir(p)


best_phase = None
corr_data = []

if make_plot:
    fig0 = plt.figure()
    plt.hold(False)

for cfile in filelist:
    
    ## wait a sufficient amount of time to ensure the file is closed
    print cfile
    #time.sleep(ts)

    if( not best_phase ):
        best_phase = getphase( cfile )

    try:
        corr = getdata( cfile, best_phase )
        corr_data.append(corr )
    except:
        print "didn't work"
        print ""
        continue
    #np.savetxt( os.path.join(path, "current_corr.txt"), [corr,] )

if make_plot:
    plt.plot(corr_data)
    plt.grid()
    plt.show()
