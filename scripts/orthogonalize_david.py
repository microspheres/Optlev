import os
import numpy as np
import matplotlib.pyplot as plt
import bead_util_david_orth as bu
import matplotlib.mlab as mlab

path = "/data/20170912/bead1_15um_QWP_NS/diag_sample_1mbar/sample_50s"
NFFT = 2**15

zdrive_file = "1mbar_zcool_G100_att_xyint_Zmod.h5"
xdrive_file = "1mbar_zcool_G100_att_xyint_ACX_synth1000mV47Hz0mVdc.h5"
data_file = "1mbar_zcool_G100_att_xyint.h5"


zd = bu.getdata(os.path.join(path, zdrive_file))
xd = bu.getdata(os.path.join(path, xdrive_file))
dd = bu.getdata(os.path.join(path, data_file))

Fs=dd["attribs"]["Fsamp"]

orth_pars = bu.calc_orthogonalize_pars( xd )

def plot_orth_spectrum(d, op):

    od = bu.orthogonalize(d['x'],d['y'],d['z'],*op)
    
    plt.figure()

    for i,c in zip([1,2,3],['x','y','z']):
        plt.subplot(3,1,i)

        p, f = mlab.psd( d[c], Fs=Fs, NFFT=NFFT )
        plt.loglog(f,p)

        p2, f2 = mlab.psd(od[i-1], Fs=Fs, NFFT=NFFT )
        plt.loglog(f2,p2)
        
plot_orth_spectrum(zd, orth_pars)

plt.show()
