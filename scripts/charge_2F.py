
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

path = r"C:\data\20170511\bead2_15um_QWP\new_sensor_feedback\charge30_freqcomb_piezo_80.0_74.9_75.4"
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


best_phase = None
corr_data = []

if make_plot:
    fig0 = plt.figure()
    plt.hold(False)

last_file = ""
while( True ):
    ## get the most recent file in the directory and calculate the correlation

    cfile = get_most_recent_file( path )
    
    ## wait a sufficient amount of time to ensure the file is closed
    print cfile
    time.sleep(ts)

    if( cfile == last_file ): 
        continue
    else:
        last_file = cfile

    ## this ensures that the file is closed before we try to read it
    time.sleep( 1 )

    if( not best_phase ):
        best_phase = getphase( cfile )

    corr = getdata( cfile, best_phase )
    corr_data.append(corr )

    np.savetxt( os.path.join(path, "current_corr.txt"), [corr,] )

    if make_plot:
        plt.plot(np.array(corr_data))
        plt.draw()
        plt.pause(0.001)
        plt.grid()
