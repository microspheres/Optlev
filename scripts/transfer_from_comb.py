import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt



#folder_meas = r"C:\data\20191210\10um\3\newpinhole\transfer"

folder_meas = r"C:\data\20200608\10um_SiO2\1\transfer1"
folder_meas = r"C:\data\20200608\10um_SiO2\1\20200617\transfer"

print folder_meas

comb = np.array([1866, 2224, 2432, 2704, 2984, 3290, 3576, 3870, 4126, 4464])*1e4/2**19


comb = 1.3*comb # this 1.3 is only for 20200608

comb[6] = comb[6]-1e4/2**19

filedata = glob.glob(folder_meas+"\*.h5")


drive_col = 3

NFFT = 2**18


def getdata(fname):
	# print "Opening file: ", fname
	## guess at file type from extension
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		#max_volt = dset.attrs['max_volt']
		#nbit = dset.attrs['nbit']
		Fs = dset.attrs['Fsamp']
                PID = dset.attrs['PID']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)

                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xin = dat[:, 0]-numpy.mean(dat[:, 0])
        xout = dat[:, 4]-numpy.mean(dat[:, 4])
        drive = dat[:, drive_col]-numpy.mean(dat[:, drive_col])

	return [xin, xout, drive, Fs]

def harmonic(f, f0, A, gamma):
    w0 = 2.*np.pi*np.abs(f0)
    w = 2.*np.pi*f
    gamma = 2.0*np.pi*gamma

    a1 = 1.*np.abs(A)
    a3 = 1.*(w0**2 - w**2)**2 + (w*gamma)**2

    s = 1.*a1/a3

    return s

def get_poits(filedata_list, comb, path):
    Xinpsd = 0.
    Xoutpsd = 0.
    Drivepsd = 0.
    for k in filedata_list:
        a = getdata(k)
        xin = a[0]
        xout = a[1]
        drive = a[2]
        Fs = a[3]

        xinpsd, freqs = matplotlib.mlab.psd(xin, Fs = Fs, NFFT = NFFT)
        xoutpsd, freqs = matplotlib.mlab.psd(xout, Fs = Fs, NFFT = NFFT)
        drivepsd, freqs = matplotlib.mlab.psd(drive, Fs = Fs, NFFT = NFFT)

        Xinpsd = xinpsd + Xinpsd
        Xoutpsd = xoutpsd + Xoutpsd
        Drivepsd = drivepsd + Drivepsd

    xoutpsd = Xoutpsd
    xinpsd = Xinpsd
    drivepsd = Drivepsd

    index = []
    for i in comb:
        b = np.where(freqs >= i)[0][0]
        index.append(b)

    drive_psd = drivepsd[index]

    xin_psd = xinpsd[index]/drive_psd
    xout_psd = xoutpsd[index]/drive_psd

    poptin, pcovin = opt.curve_fit(harmonic, freqs[index], xin_psd, p0 = [59.2, 1e9, 1.6])
    poptout, pcovout = opt.curve_fit(harmonic, freqs[index], xout_psd, p0 = [59.2, 1e9, 1.6])

    xallpsd = (xinpsd + xoutpsd)
    xall_psd = xallpsd[index]/drive_psd

    poptall, pcovall = opt.curve_fit(harmonic, freqs[index], xall_psd, p0 = [59.2, 1e9, 1.6])
    
    
    plt.figure()
    plt.loglog(freqs, drivepsd)
    plt.loglog(freqs[index], drivepsd[index], "r.")

    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.loglog(freqs[index], xin_psd, "r.")
    plt.loglog(freqs[index], xout_psd, "b.")
    plt.loglog(freqs[index], xall_psd, ".")
    plt.loglog(freqs, harmonic(freqs, *poptin), "r-")
    plt.loglog(freqs, harmonic(freqs, *poptout), "b-")
    plt.loglog(freqs, harmonic(freqs, *poptall))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Transfer function")
    plt.tight_layout()

    gammain = poptin[2]
    gammaout = poptout[2]

    sin2 = pcovin[2][2]
    sout2 = pcovout[2][2]
    s = 1./(1./sin2 + 1./sout2)
    
    gamma_combined = 1.*s*(gammain/sin2 + gammaout/sout2)

    print "f0out = ", poptout[0]
    print "f0in = ", poptin[0]
    print "gammaout = ", gammaout
    print "gammain = ", gammain
    print ""
    print "errors in"
    print "f0_err = ", np.sqrt(pcovin[0][0])
    print "gamma_err = ", np.sqrt(pcovin[2][2])
    print ""
    print "errors out"
    print "f0_err = ", np.sqrt(pcovout[0][0])
    print "gamma_err = ", np.sqrt(pcovout[2][2])
    print ""
    print "combined inloop and outloop"
    print "f0 = ", poptall[0]
    print "gamma", poptall[2]
    print "f0_err = ", np.sqrt(pcovall[0][0])
    print "gamma_err = ", np.sqrt(pcovall[2][2])

    gamma = [gammain, gammaout, gamma_combined]
    name = str(folder_meas) + "\\" +"gammas.npy"
    np.save(name, gamma)
    return []


get_poits(filedata, comb, folder_meas)
plt.show()
