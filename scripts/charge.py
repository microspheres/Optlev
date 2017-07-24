## load all files in a directory and plot the correlation of the response
## with the drive signal versus time

import matplotlib.pyplot as plt
import bead_util as bu
import numpy as np
import os, time

path = r"C:\data\20170717\bead15_15um_QWP\charge6"
ts = 1.

fdrive = 31.
make_plot = True
debugging = False


def getphase(fname):
    print "Getting phase from: ", fname
    dat, attribs, cf = bu.getdata(os.path.join(path, fname))
    fsamp = attribs["Fsamp"]
    xdat = dat[:, bu.xi]

    xdat = np.append(xdat, np.zeros(int(fsamp / fdrive)))
    corr2 = np.correlate(xdat, dat[:, bu.drive])
    maxv = np.argmax(corr2)

    cf.close()

    print maxv
    return maxv


def getdata(fname, maxv):
    print "Processing ", fname
    dat, attribs, cf = bu.getdata(os.path.join(path, fname))

    fsamp = attribs["Fsamp"]

    xdat = dat[:, bu.xi]

    ## zero pad one cycle
    MinusDC = dat[:, bu.drive]
    corr_full = bu.corr_func(MinusDC - np.median(MinusDC), xdat, fsamp, fdrive)

    if debugging:
        plt.figure()
        plt.plot(xdat)
        plt.plot(MinusDC)
        plt.show()

    return corr_full[0], np.max(corr_full)


def get_most_recent_file(p):
    filelist = bu.time_ordered_file_list(p)
    return filelist[-1]


best_phase = None
corr_data = []

if make_plot:
    fig0 = plt.figure()
    plt.hold(False)

last_file = ""
while (True):
    ## get the most recent file in the directory and calculate the correlation

    cfile = get_most_recent_file(path)

    ## wait a sufficient amount of time to ensure the file is closed
    print cfile
    time.sleep(ts)

    if (cfile == last_file):
        continue
    else:
        last_file = cfile

    ## this ensures that the file is closed before we try to read it
    time.sleep(1)

    if (not best_phase):
        best_phase = getphase(cfile)

    corr = getdata(cfile, best_phase)
    corr_data.append(corr)

    np.savetxt(os.path.join(path, "current_corr.txt"), [corr, ])

    if make_plot:
        plt.plot(np.array(corr_data))
        plt.draw()
        plt.pause(0.001)
        plt.grid()
