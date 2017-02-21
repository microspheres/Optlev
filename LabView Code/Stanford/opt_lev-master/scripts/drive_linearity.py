
## load all files in a directory and plot the correlation of the resonse
## with the drive signal versus time

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, glob
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle
import matplotlib.gridspec as gridspec

path = "/data/20140617/Bead3/recharge_vramphi"
## path to directory containing charge steps, used to calibrate phase and 
## scaling.  leave empty to use data path
cal_path = "/data/20140617/Bead3/chargelp"

## path to save plots and processed files (make it if it doesn't exist)
outpath = "/home/dcmoore/analysis" + path[5:]
if( not os.path.isdir( outpath ) ):
    os.makedirs(outpath)

#Htot = np.load(os.path.join("/home/dcmoore/analysis" + cal_path[5:], "trans_func.npy") )

reprocessfile = False
plot_angle = False
plot_phase = False
remove_laser_noise = False
remove_outliers = True
plot_flashes = False
ref_file = 0 ## index of file to calculate angle and phase for

file_start = 0

scale_fac = 1. ##1./0.00156 * 1/63.
scale_file = 1.

amp_gain = 200. ## gain to use for files in path
amp_gain_cal = 1.  ## gain to use for files in cal_path

fsamp = 5000.
fdrive = 41.
fref = 1027
NFFT = 2**14
phaselen = int(fsamp/fdrive) #number of samples used to find phase
plot_scale = 1. ## scaling of corr coeff to units of electrons
plot_offset = 1.
data_columns = [0, 1] ## column to calculate the correlation against
drive_column = -1
laser_column = 3


b, a = sp.butter(3, [2.*(fdrive-1)/fsamp, 2.*(fdrive+1)/fsamp ], btype = 'bandpass')
boff, aoff = sp.butter(3, 2.*(fdrive-10)/fsamp, btype = 'lowpass')

def rotate_data(x, y, ang):
    c, s = np.cos(ang), np.sin(ang)
    return c*x - s*y, s*x + c*y

def getangle(fname):
        print "Getting angle from: ", fname 
        num_angs = 100
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))
        pow_arr = np.zeros((num_angs,2))
        ang_list = np.linspace(-np.pi/2.0, np.pi/2.0, num_angs)
        for i,ang in enumerate(ang_list):
            rot_x, rot_y = rotate_data(dat[:,data_columns[0]], dat[:,data_columns[1]], ang)
            pow_arr[i, :] = [np.std(rot_x), np.std(rot_y)]
        
        best_ang = ang_list[ np.argmax(pow_arr[:,0]) ]
        print "Best angle [deg]: %f" % (best_ang*180/np.pi)

        cf.close()

        if(plot_angle):
            plt.figure()
            plt.plot(ang_list, pow_arr[:,0], label='x')
            plt.plot(ang_list, pow_arr[:,1], label='y')
            plt.xlabel("Rotation angle")
            plt.ylabel("RMS at drive freq.")
            plt.legend()
            
            ## also plot rotated time stream
            rot_x, rot_y = rotate_data(dat[:,data_columns[0]], dat[:,data_columns[1]], best_ang)
            plt.figure()
            plt.plot(rot_x)
            plt.plot(rot_y)
            plt.plot(dat[:, drive_column] * np.max(rot_x)/np.max(dat[:,drive_column]))
            plt.show()
        
        

        return best_ang

def getphase(fname, ang):
        print "Getting phase from: ", fname 
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))
        xdat, ydat = rotate_data(dat[:,data_columns[0]], dat[:,data_columns[1]], ang)
        #xdat = sp.filtfilt(b, a, xdat)
        xdat = np.append(xdat, np.zeros( fsamp/fdrive ))
        corr2 = np.correlate(xdat,dat[:,drive_column])
        maxv = np.argmax(corr2) 

        cf.close()

        if(plot_phase):
            plt.figure()
            plt.plot( corr2 )
            plt.figure()
            xdat_filt = sp.filtfilt(b,a,xdat)
            drive_filt = sp.filtfilt(b,a,dat[:,drive_column])
            plt.plot( xdat_filt/np.max( xdat_filt ), label='x')
            plt.plot( drive_filt/np.max( drive_filt ), label='drive' )
            plt.legend()
            plt.show()

        print maxv
        return maxv


def getdata(fname, maxv, ang, gain):

	print "Processing ", fname
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))

        ## make sure file opened correctly
        if( len(dat) == 0 ):
            return {}

        dat[:, drive_column] *= gain
        if( len(attribs) > 0 ):
            fsamp = attribs["Fsamp"]
            drive_amplitude = attribs["drive_amplitude"]

            
        xdat, ydat = rotate_data(dat[:,data_columns[0]], dat[:,data_columns[1]], ang)

        drive_amp = np.sqrt(2)*np.std(dat[:,drive_column])

        if( remove_laser_noise ):
            laser_good = bu.laser_reject(dat[:, laser_column], 60., 90., 4e-6, 100, fsamp, False)
        else:
            laser_good = np.ones_like(dat[:, laser_column]) > 0

        #df = np.fft.rfft( dat[:,drive_column] )
        #drive_pred = np.fft.irfft( df*Htot )

        corr_full = bu.corr_func(dat[:,drive_column], xdat, fsamp, fdrive, good_pts=laser_good)
        #corr_full = bu.corr_func(drive_pred, xdat, fsamp, fdrive, good_pts=laser_good)

        #corr = corr_full[ maxv ]
        corr = corr_full[ 0 ]

        corr_max = np.max(corr_full)
        corr_max_pos = np.argmax(corr_full)
        xpsd, freqs = matplotlib.mlab.psd(xdat, Fs = fsamp, NFFT = NFFT) 
        #ypsd, freqs = matplotlib.mlab.psd(ydat, Fs = fsamp, NFFT = NFFT) 
        max_bin = np.argmin( np.abs( freqs - fdrive ) )
        ref_bin = np.argmin( np.abs( freqs - fref ) )

        ## also correlate signal with drive squared
        dsq = dat[:,drive_column]**2
        dsq -= np.mean(dsq)
        sq_amp = np.sqrt(2)*np.std( dsq )
        ## only normalize by one factor of the squared amplitude
        corr_sq_full = bu.corr_func(dsq*sq_amp, xdat, fsamp, fdrive)
        corr_sq_max = np.max(corr_sq_full)
        corr_sq_max_pos = np.argmax(corr_sq_full)

        xoff = sp.filtfilt(boff, aoff, xdat)

        if(False):
            plt.figure()
            plt.plot( xdat )
            plt.plot( dat[:, drive_column] )

            plt.figure()
            plt.plot( corr_full )
            plt.show()

        ctime = attribs["time"]

        ## is this a calibration file?
        cdir,_ = os.path.split(fname)
        is_cal = cdir == cal_path

        curr_scale = 1.0
        ## make a dictionary containing the various calculations
        out_dict = {"corr_t0": corr,
                    "max_corr": [corr_max, corr_max_pos],
                    "max_corr_sq": [corr_sq_max, corr_sq_max_pos],
                    "psd": np.sqrt(xpsd[max_bin]),
                    "ref_psd": np.sqrt(xpsd[ref_bin]),
                    "temps": attribs["temps"],
                    "time": bu.labview_time_to_datetime(ctime),
                    "num_flashes": attribs["num_flashes"],
                    "is_cal": is_cal,
                    "drive_amp": drive_amp}

        cf.close()
        return out_dict

if reprocessfile:

  init_list = glob.glob(path + "/*.h5")
  files = sorted(init_list, key = bu.find_str)

  if(cal_path):
      cal_list = glob.glob(cal_path + "/*.h5")
      cal_files = sorted( cal_list, key = bu.find_str )
      files = zip(cal_files[:-1],np.zeros(len(cal_files[:-1]))+amp_gain_cal) \
              + zip(files[:-1],np.zeros(len(files[:-1]))+amp_gain)
  else:
      files = zip(files[:-1],np.zeros(len(files[:-1]))+amp_gain)      
      

  ang = 0 ##getangle(files[ref_file])
  phase = getphase(files[ref_file][0], ang)
  corrs_dict = {}
  for f,gain in files[file_start:]:
    curr_dict = getdata(f, phase, ang, gain)

    for k in curr_dict.keys():
        if k in corrs_dict:
            corrs_dict[k].append( curr_dict[k] )
        else:
            corrs_dict[k] = [curr_dict[k],]
    
  of = open(os.path.join(outpath, "processed.pkl"), "wb")
  pickle.dump( corrs_dict, of )
  of.close()
else:
  of = open(os.path.join(outpath, "processed.pkl"), "rb")
  corrs_dict = pickle.load( of )
  of.close()

## if a calibration data set is defined and the scale factor is 1,
## then try to calculate the scale factor from the calibration
is_cal = np.array( corrs_dict["is_cal"] )
if( np.sum(is_cal) > 0 and scale_fac == 1.):
    cal_dat = np.array(corrs_dict["corr_t0"])[is_cal]
    ## take a guess at the step size
    step_vals = np.abs( np.diff( cal_dat ) )
    step_guess = np.median( step_vals[ step_vals > 3*np.std(step_vals)] )
    ## only keep non-zero points (assuming sig-to-noise > 5)
    cal_dat = cal_dat[cal_dat > 0.2*step_guess]
    def scale_resid( s ):
        return np.sum( (cal_dat/s - np.round(cal_dat/s))**2  )
    ## do manual search for best scale fac
    slist = np.linspace(step_guess/1.2, step_guess*1.2, 1e4)
    scale_fac = 1./slist[np.argmin( map(scale_resid, slist) ) ]
    print "Calibration: guess, best_fit: ", 1./step_guess, scale_fac
    
## first plot the variation versus time
dates = matplotlib.dates.date2num(corrs_dict["time"])
corr_t0 = np.array(corrs_dict["corr_t0"])*scale_fac
max_corr = np.array(corrs_dict["max_corr"])[:,0]*scale_fac
max_corr_sq = np.array(corrs_dict["max_corr_sq"])[:,0]*scale_fac
best_phase = np.array(corrs_dict["max_corr"])[:,1]
psd = np.array(corrs_dict["psd"])*scale_fac
ref_psd = np.array(corrs_dict["ref_psd"])*scale_fac
temp1 = np.array(corrs_dict["temps"])[:,0]
temp2 = np.array(corrs_dict["temps"])[:,1]
num_flashes = np.array(corrs_dict["num_flashes"])
drive_amp = np.array(corrs_dict["drive_amp"])

fig=plt.figure() 
gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])
plt.subplot(gs[0])
#plt.plot_date(dates, corr_t0, 'r.', label="Max corr")
x = drive_amp[np.logical_not(is_cal)]
y = x*corr_t0[np.logical_not(is_cal)]
plt.plot( x, y, 'k.', label='Measured response' )
xl = [0, x.max()*1.05]

gpts = x < 100
p = np.polyfit( x[gpts], y[gpts], 1)
xx = np.linspace( xl[0], xl[1], 1e2)

plt.plot(xx, np.polyval(p, xx), 'r', linewidth=1.5, label='Linear fit')
#plt.xlabel("Drive amplitude [V]")
plt.ylabel("Drive response, q*V [e V]")
plt.xlim(xl)
plt.legend(numpoints=1)

plt.subplot(gs[1])
plt.plot(x, y-np.polyval(p, x), 'k.', linewidth=1.5)
plt.plot(xx, np.zeros_like(xx), 'r', linewidth=1.5)
plt.ylabel("Data - Fit")
plt.xlabel("Drive amplitude [V]")
plt.ylim([-20,20])
plt.xlim(xl)

fig.set_size_inches(8,8)
plt.savefig( os.path.join(outpath, "drive_linearity.pdf") )

plt.show()

