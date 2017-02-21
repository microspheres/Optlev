import os, glob, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt

#########################################################

path = r"C:\Data\20150826\Bead1\electrode_sweep4"
#path = r"/data/20150825/Bead2/junk"
name = "urmbar_xyzcool_*.h5"

## columns containing electrode signals (skip ch10 since bad signal)
electrode_cols = [7,8,9,11,12,13,14]

## columns to calculate the correlation against
data_cols = [0,1,2]

plot_freqs = False ## plot recorded frequencies

#########################################################

def sort_fun(s):
    vstr = re.findall("-?\d+mVdc.h5", s)[0]
    return float( vstr[:-7] )

flist = sorted( glob.glob( os.path.join( path, name ) ), key = sort_fun )

col_list = ['k','b','r','g','m','c','y']

drive_freq_list = []
corr_mat = np.zeros( (len(flist), len(data_cols), len(electrode_cols), 2 ) )
dc_list = np.zeros( len(flist) )
for fidx, f in enumerate(flist):

    dat, attribs, cf = bu.getdata( f )

    if( len(dat) == 0 ): 
        print "Warning, couldn't get data for: ", f

    fsamp = attribs["Fsamp"]

    print "Working on file: ", f

    if len(drive_freq_list) == 0:
        ## get the frequency for each drive channel
        if(plot_freqs): plt.figure()
        for j,elec in enumerate(electrode_cols):
            
            cpsd = np.abs(np.fft.rfft( dat[:, elec]-np.mean(dat[:, elec]) ))**2
            cfreqs = np.fft.rfftfreq( len( dat[:, elec] ), d=1./fsamp )
            cidx = np.argmax( cpsd )
            curr_max = [ cfreqs[cidx], cidx ]

            drive_freq_list.append( curr_max )

            if( plot_freqs ):
                plt.loglog( cfreqs, cpsd, color=col_list[j] )
                plt.plot( cfreqs[cidx], cpsd[cidx], 'o', color=col_list[j], linewidth=1.5 )
                
        if(plot_freqs): plt.show()
        drive_freq_list = np.array( drive_freq_list )

        print "Drive freqs are:  ", drive_freq_list[:,0]

    for didx, dcol in enumerate(data_cols):
        for eidx, ecol in enumerate(electrode_cols):
        
            corr_full = bu.corr_func( dat[:,ecol], dat[:, dcol], fsamp, drive_freq_list[eidx][0] )
            dc_val = np.mean( dat[:, ecol] )

            corr_mat[fidx, didx, eidx, :] = [dc_val, corr_full[0]]


## now plot total correlation
for eidx, ecol in enumerate(electrode_cols):
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot( corr_mat[:, 0, eidx, 0], corr_mat[:, 0, eidx, 1], color=col_list[eidx] )
    plt.title("Electrode %d, X" % eidx)

    plt.subplot(3,1,2)
    plt.plot( corr_mat[:, 1, eidx, 0], corr_mat[:, 1, eidx, 1], color=col_list[eidx] )
    plt.title("Electrode %d, Y" % eidx)

    plt.subplot(3,1,3)
    plt.plot( corr_mat[:, 2, eidx, 0], corr_mat[:, 1, eidx, 1], color=col_list[eidx] )
    plt.title("Electrode %d, Z" % eidx)

plt.show()

