import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt
from scipy.signal import butter, lfilter, filtfilt


folder_calibration = r"C:\data\20191122\10um\2\calibration1p"

folder_meas = r"C:\data\20191122\10um\2\acceleration2"

file_list = glob.glob(folder_meas+"\*.h5")

drive_col = 3

NFFT = 2**18

def get_v_to_m_and_fressonance(folder_meas):
    namein = str(folder_meas) + r"\v2tom2_in.npy"
    nameout = str(folder_meas) + r"\v2tom2_out.npy"
    Lin = np.sqrt(np.load(namein))
    Lout = np.sqrt(np.load(nameout))
    namefress = str(folder_meas) + r"\info_outloop.npy"
    fress = np.load(namefress)[7]
    return [Lin, Lout, fress]

v_to_m_in, v_to_m_out, fres = get_v_to_m_and_fressonance(folder_meas)

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

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

def get_drive_frequency_timestream(folder_calibration): # return freq, freq_arg, amplitude of the drive and sphere x motion in the freq band.
    file_list = glob.glob(folder_calibration+"\*.h5")

    i = file_list[0]
    a = getdata(i)

    drive = a[2]
    Fs = a[3]
    xin = a[0]
    xout = a[1]

    fft_drive = np.fft.rfft(drive, drive.size)
    drive_psd = fft_drive*np.conjugate(fft_drive)
    freq = np.fft.rfftfreq(drive.size, 1./Fs)
    
    f0arg = np.argmax(drive_psd)
    f0 = freq[f0arg]

    
    # plt.figure()
    # plt.loglog(freq, drive_psd)
    # plt.loglog(freq, fft_x*np.conjugate(fft_x))
    # plt.loglog(freq[f0arg], drive_psd[f0arg], "ro")
    # plt.show()
    return [f0, f0arg, drive, xin, xout, Fs, freq]

def template_time_stream(folder_calibration):
    a = get_drive_frequency_timestream(folder_calibration)
    Fs = a[5]
    drive = a[2]
    xin = a[3]
    xout = a[4]
    arg0 = a[1]
    f = a[0]
    freq0 = a[6]

    df = 0.2

    # fft_drive = np.fft.rfft(drive, drive.size)
    # drivef = butter_bandpass_filter(drive, f - 0.5, f + 0.5, Fs, 3)
    # fft_drivef = np.fft.rfft(drivef, drive.size)

    # fft_x = np.fft.rfft(x, drive.size)
    xfin = butter_bandpass_filter(xin, f - df, f + df, Fs, 2)
    xfout = butter_bandpass_filter(xout, f - df, f + df, Fs, 2)
    # fft_xf = np.fft.rfft(xf, drive.size)

    # plt.figure()
    # # plt.loglog(freq, fft_drive*np.conjugate(fft_drive))
    # # plt.loglog(freq, fft_drivef*np.conjugate(fft_drivef))

    # plt.loglog(freq, fft_x*np.conjugate(fft_x))
    # plt.loglog(freq, fft_xf*np.conjugate(fft_xf))
    # plt.show()
    
    return [xfin, xfout, f, df, Fs]


def corr(folder_meas, v_to_m_in, v_to_m_out, folder_calibration, fres, cross_spectra):
    a = template_time_stream(folder_calibration)
    Fs = a[4]
    df = a[3]
    df = 0.5
    f = a[2]
    x_template_in = a[0]
    x_template_out = a[1]

    Cin = []
    Cout = []

    Nin = np.sum(x_template_in*x_template_in)/(0.5*len(x_template_in))
    Nout = np.sum(x_template_out*x_template_out)/(0.5*len(x_template_out))

    file_list = glob.glob(folder_meas+"\*.h5")
    file_list = list_file_time_order(file_list)
    L = len(file_list)

    ac =  (2.*np.pi*fres)**2 - (2.*np.pi*f)**2

    Cross = []
    Psd = []

    j = 0
    for i in file_list:
        xin = getdata(i)[0]
        xout = getdata(i)[1]

        if cross_spectra:
            frequency, Pxy = sp.csd(xin, xout, Fs, nperseg=NFFT)
            frequency, Pxx = sp.csd(xin, xin, Fs, nperseg=NFFT)
            frequency, Pyy = sp.csd(xout, xout, Fs, nperseg=NFFT)
            cross = np.real( Pxy )**2 / (Pxx*Pyy)
            cross = np.sqrt(cross)
            Cross.append(cross)
            psd = np.abs(np.real( Pxy ))*(v_to_m_out*v_to_m_in*(ac**2))/((9.8e-6)**2)
            Psd.append(np.sqrt(psd))
        
        j = j+1
        if j%30 == 0:
            print 1.*j/L

        xfin = butter_bandpass_filter(xin, f - df, f + df, Fs, 2)
        cin = np.sum(xfin*x_template_in)/(0.5*len(xfin))
        cin = cin/np.sqrt(Nin)
        Cin.append(cin)

        xfout = butter_bandpass_filter(xout, f - df, f + df, Fs, 2)
        cout = np.sum(xfout*x_template_out)/(0.5*len(xfout))
        cout = cout/np.sqrt(Nout)
        Cout.append(cout) 

    Cin = np.array(Cin)/9.8e-9
    Cin = Cin*v_to_m_in*ac

    Cout = np.array(Cout)/9.8e-9
    Cout = Cout*v_to_m_out*ac
    
    return [Cin, Cout, [Cross, frequency], Psd]


def gauss(x,a,b,c):
    g = c*np.exp(-0.5*((x-a)/b)**2)
    return g

def histo(c, bins):
    h,b = np.histogram(c, bins = bins)
    bc = np.diff(b)/2 + b[:-1]
    return [h, bc]


def gauss_filter(c1, c2, sigma, cross, psd):
    w1 = np.std(c1)
    w2 = np.std(c2)
    cnew1 = []
    cnew2 = []
    Cross = []
    Psd = []
    for i in range(len(c1)):
        if np.abs(c1[i]) <= np.abs(1.*sigma*w1) and np.abs(c2[i]) <= np.abs(1.*sigma*w2):
            cnew1.append(c1[i])
            cnew2.append(c2[i])
            Cross.append(cross[i])
            Psd.append(psd[i])
    
    cnew1 = np.array(cnew1)
    cnew2 = np.array(cnew2)
    return [cnew1, cnew2, Cross, Psd]


cin, cout, cross, Psd = corr(folder_meas, v_to_m_in, v_to_m_out, folder_calibration, fres, True)
frequency = cross[1]
cross = cross[0]
bins = 13

#####
for i in range(4):
    cin, cout, cross, psd = gauss_filter(cin, cout, 3, cross, Psd)
#####

hin, bcin = histo(cin, bins)
hout, bcout = histo(cout, bins)

poptin, pcovin = opt.curve_fit(gauss, bcin, hin)
poptout, pcovout = opt.curve_fit(gauss, bcout, hout)


vin = np.abs(poptin[1]**2)
vout = np.abs(poptout[2]**2)


v_combined = 1./(1./vin + 1./vout)

c = (1.*cin/vin + 1.*cout/vout)*v_combined

h, bc = histo(c, bins)
popt, pcov = opt.curve_fit(gauss, bc, h)


space = np.linspace(2*np.min(bc),2*np.max(bc), 1000)

acin = str(poptin[0]) + r"$\pm$" + str(np.sqrt(pcovin[0][0]))
acout = str(poptout[0]) + r"$\pm$" + str(np.sqrt(pcovout[0][0]))
ac = str(popt[0]) + r"$\pm$" + str(np.sqrt(pcov[0][0]))

print "acc in nano-g:"
print "accin =", acin
print "accout =", acout
print "acc =", ac

plt.figure()
plt.plot(bcin, hin, "r.")
plt.plot(bcout, hout, "b.")
plt.plot(bc, h, "g.")
plt.plot(space, gauss(space, *poptin), "r-")
plt.plot(space, gauss(space, *poptout), "b-")
plt.plot(space, gauss(space, *popt), "g-")
plt.tight_layout(pad = 0)

x = range(len(cin))
import matplotlib.cm as cm
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(x, cin, "r.")
plt.subplot(3, 1, 2)
plt.plot(x, cout, "b.")
plt.subplot(3, 1, 3)
plt.plot(x, c, "g.")
plt.tight_layout(pad = 0)

space = np.linspace(np.min(cin), np.max(cin), 1000)

plt.figure()
plt.plot(space, space, "k--")
plt.scatter(cin, cout, c = x, cmap = cm.viridis)
plt.xlabel("inloop [ng]")
plt.ylabel("outloop [ng]")
plt.colorbar()
plt.tight_layout(pad = 0)


nc = np.zeros(len(cross[0]))
for j in cross:
    nc = nc + j
nc = nc/len(cross)
plt.figure()
plt.loglog(frequency, nc)
plt.xlabel("freq [Hz]")
plt.ylabel("coherence")
plt.xlim(10,100)
plt.ylim(0.5,1)

npsd = np.zeros(len(psd[0]))
for j in psd:
    npsd = npsd + j
npsd = npsd/len(psd)
plt.figure()
plt.loglog(frequency, npsd)
plt.xlabel("freq [Hz]")
plt.ylabel("ug/sqrt(Hz)")
plt.xlim(10,100)

plt.show()
