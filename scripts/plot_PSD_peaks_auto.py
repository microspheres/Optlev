from plot_PSD_peaks import plot_peaks2Fernando
from charge import get_most_recent_file
from correlation import outputThetaPosition
from plot_corr_peaks import getdata_x_d, corr_aux
import matplotlib.pyplot as plt
import numpy as np
import time

path = "/data/20170717/bead15_15um_QWP/dipole4_Z"

ts = 1

theta, corr = ([] for i in range(2))
y_or_z = ""
last_file = ""

plt.figure()
plt.hold(False)
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

    x, d, d2 = getdata_x_d(cfile)
    corra, corr2a = corr_aux(d2, d, x, 1, 0)
    corr.append(corra)
    t, y_or_z = outputThetaPosition(cfile, y_or_z)
    theta.append(t)
    corr.append(np.correlate(x, d))

    plt.plot(theta, corr, 'o')
    plt.draw()
    plt.pause(0.001)
    plt.grid()
