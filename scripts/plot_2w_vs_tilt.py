import numpy, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
from scipy.optimize import curve_fit

X = []
Y = []
for i in range(19):
    path_sphere = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\electrode_dist_fartherN2\tilt"+"%d"%(i+1)
    file_sphere = glob.glob(path_sphere+"\*arb*.txt")
    sphere = np.loadtxt(file_sphere[0])
    x = sphere[0]
    y = sphere[1]
    X.append(x)
    Y.append(y)
    
plt.figure()
for i in range(len(X)):
    plt.semilogy(X[i], Y[i])
plt.xlim(45,100)
plt.show()
