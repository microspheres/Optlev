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

path = r"D:\Data\20150528\Bead1\angle"
ts = 10.
best_phase = 2

fdrive = 6.
make_plot = True

data_columns = [2, 1] ## column to calculate the correlation against
drive_column = -1 ## column containing drive signal
mod_column = 3

def ffn(x, p0, p1, p2):
        return p0*np.sin( 2*np.pi*p1*x + p2) 


def getmodulation(dat, plt_fit = False):
        b,a = sp.butter(3,0.1)
        cdrive = np.abs(dat[:,mod_column]-np.mean(dat[:,mod_column]))
        cdrive = sp.filtfilt( b, a, cdrive )
        cdrive -= np.mean(cdrive)
        ## fit to a sin
        xx = np.arange(len(cdrive))
        spars = [np.std(cdrive)*2, 6.1e-4, 0]
        bf, bc = opt.curve_fit( ffn, xx, cdrive, p0=spars )
        npts_per_cycle = 1./bf[1]
        init_phase = (bf[2]/(2*np.pi) + 0.25 )*npts_per_cycle
        print npts_per_cycle, init_phase
        if False:
                plt.figure()
                plt.hold(True)
                plt.plot(xx[::4] ,dat[::4,3]-np.mean(dat[:,3]))
                #plt.show()
                plt.plot(xx[::4], cdrive[::4], 'k.' )
                #plt.show()
                plt.plot(xx[::4], ffn(xx[::4], bf[0], bf[1], bf[2]), 'r' )
                plt.show()
                print "hello world"
        return ffn(xx, bf[0], bf[1], bf[2])/np.abs(bf[0])


def getphase(fname):
        print "Getting phase from: ", fname 
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))
        fsamp = attribs["Fsamp"]
        xdat = dat[:,data_columns[0]]

        xdat = np.append(xdat, np.zeros( int(fsamp/fdrive) ))
        corr2 = np.correlate(xdat,getmodulation(dat, plt_fit = True))
        maxv = np.argmax(corr2) 

        cf.close()

        print maxv
        return maxv


def getdata(fname, maxv):

	print "Processing ", fname
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))

        if( len(attribs) > 0 ):
            fsamp = attribs["Fsamp"]

        xdat = dat[:,data_columns[0]]

        lentrace = len(xdat)
        ## zero pad one cycle
        corr_full = bu.corr_func(getmodulation(dat), xdat, fsamp, fdrive)
        corrzero = np.sum(getmodulation(dat)*xdat)

        return corr_full[maxv], np.max(corr_full)
        #return corrzero, np.max(corr_full)
        

def get_most_recent_file(p):

    ## only consider single frequency files, not chirps
    filelist = glob.glob(os.path.join(p,"*Hz*.h5"))  ##os.listdir(p)
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
    time.sleep(1)

    if( cfile == last_file ): 
        continue
    else:
        last_file = cfile

    ## this ensures that the file is closed before we try to read it
    time.sleep( ts )

    if( not best_phase ):
        best_phase = getphase( cfile )
        print "Found phase offset: ", best_phase

    corr = getdata( cfile, best_phase )
    corr_data.append(corr )

    np.savetxt( os.path.join(path, "current_corr.txt"), [corr,] )

    if make_plot:
        plt.plot(np.array(corr_data))
        plt.draw()
        plt.pause(0.001)

    
