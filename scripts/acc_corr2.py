import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import scipy.optimize as opt
from scipy.signal import butter, lfilter, filtfilt
def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist


folder_calibration = r"C:\data\20191210\10um\3\newpinhole\calibration1e"

folder_meas = r"C:\data\20191210\10um\3\newpinhole\acceleration2"

file_list_meas = glob.glob(folder_meas+"\*.h5")
file_list_meas = list_file_time_order(file_list_meas)[0:840]
#file_list_meas = list_file_time_order(file_list_meas)[0:50]

use_transfer_gammas = True
if use_transfer_gammas:
    folder_trasfer = r"C:\data\20191210\10um\3\newpinhole\transfer"

use_drive_freq = False # allow any choice of freq for correlation, if false the frequency can be any

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

def get_gammas(folder_trasfer):
    name = str(folder_trasfer) + r"\gammas.npy"
    g = np.load(name)
    return g

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

    if use_drive_freq:
        freq, drive_psd = sp.csd(drive, drive, Fs, nperseg=NFFT)
    
        f0arg = np.argmax(drive_psd)
        f0 = freq[f0arg]
    else:
        time = range(0, 2**19)/Fs
        new = np.sin(time*2.*np.pi*2422*Fs/2**19) # f0 = N*Fs/2**19
        new = new - np.mean(new)
        freq, drive_psd = sp.csd(new, new, Fs, nperseg=NFFT)
        f0arg = np.argmax(drive_psd)
        f0 = freq[f0arg]
        drive = new

    print "frequency", f0
    
    # plt.figure()
    # plt.loglog(freq, drive_psd)
    # plt.loglog(freq[f0arg], drive_psd[f0arg], "ro")
    # plt.show()
    return [f0, f0arg, drive, xin, xout, Fs, freq]
    #return [f0, f0arg, drive, drive, drive, Fs, freq] # the return can be drive instead of xin or xout, at the amplitude of the template gets normalized.

def template_time_stream(folder_calibration):
    a = get_drive_frequency_timestream(folder_calibration)
    Fs = a[5]
    drive = a[2]
    xin = a[3]
    xout = a[4]
    arg0 = a[1]
    f = a[0]
    freq0 = a[6]
    
    return [xin, xout, f, arg0, Fs]


def corr(file_list_meas, v_to_m_in, v_to_m_out, folder_calibration, fres, cross_spectra):
    a = template_time_stream(folder_calibration)
    Fs = a[4]
    arg = a[3]
    f = a[2]
    x_template_in = a[0]
    x_template_out = a[1]

    Cin = []
    Cout = []
    
    frequency, Nin = sp.csd(x_template_in, x_template_in, Fs, nperseg=NFFT, scaling = "spectrum")
    frequency, Nout = sp.csd(x_template_out, x_template_out, Fs, nperseg=NFFT, scaling = "spectrum")

    L = len(file_list_meas)

    index0 = np.where(frequency > 1)[0][0]
    index1 = np.where(frequency > 200)[0][0]

    if use_transfer_gammas:
        gammain, gammaout, g = get_gammas(folder_trasfer)
        usef = frequency
    else:
        gammain, gammaout, g = [0., 0., 0.]
        usef = f

    acin_psd = np.sqrt( ((2.*np.pi*fres)**2 - (2.*np.pi*usef)**2)**2 + ( ((2.*np.pi*gammain)*(2.*np.pi*usef)) )**2 )
    acout_psd = np.sqrt( ((2.*np.pi*fres)**2 - (2.*np.pi*usef)**2)**2 + ( ((2.*np.pi*gammaout)*(2.*np.pi*usef)) )**2 )
    ac_comb_psd = np.sqrt( ((2.*np.pi*fres)**2 - (2.*np.pi*usef)**2)**2 + ( ((2.*np.pi*g)*(2.*np.pi*usef)) )**2 )

    acin_psd = acin_psd[index0:index1]
    acout_psd = acout_psd[index0:index1]
    ac_comb_psd = ac_comb_psd[index0:index1]

    ac_corr =  np.sqrt( ((2.*np.pi*fres)**2 - (2.*np.pi*f)**2)**2 + ( ((2.*np.pi*g)*(2.*np.pi*f)) )**2 ) # the corr only needs the transfer func at a specific frequency.

    Cross = []
    Psd = []
    Psdin = []
    Psdout = []

    j = 0
    for i in file_list_meas:
        xin = getdata(i)[0]
        xout = getdata(i)[1]

        frequency, Pxy = sp.csd(xin, xout, Fs, nperseg=NFFT)
        frequency, Pxx = sp.csd(xin, xin, Fs, nperseg=NFFT)
        frequency, Pyy = sp.csd(xout, xout, Fs, nperseg=NFFT)
        cross = np.real( Pxy[index0:index1] )**2 / (Pxx[index0:index1]*Pyy[index0:index1])
        cross = np.sqrt(cross)
        Cross.append(cross)
        psd = np.abs(np.real( Pxy[index0:index1] ))*(v_to_m_out*v_to_m_in*(ac_comb_psd**2))/((9.8e-6)**2)
        Psd.append(psd)
        psdin = Pxx[index0:index1]*(v_to_m_in*v_to_m_in*(acin_psd**2))/((9.8e-6)**2)
        Psdin.append(psdin)
        psdout = Pyy[index0:index1]*(v_to_m_out*v_to_m_out*(acout_psd**2))/((9.8e-6)**2)
        Psdout.append(psdout)
        # plt.figure()
        # plt.loglog(frequency, psd)
        # plt.loglog(frequency, psdin)
        # plt.loglog(frequency, psdout)
        # plt.show()
        
        j = j+1
        if j%30 == 0:
            print 1.*j/L

        frequency, Pxyin = sp.csd(xin, x_template_in, Fs, nperseg=NFFT, scaling = "spectrum")
        cin = np.sum( np.real(Pxyin[arg-1:arg+1]) )
        cin = cin/np.sum( np.sqrt(Nin[arg-1:arg+1]) )
        Cin.append(cin)

        frequency, Pxyout = sp.csd(xout, x_template_out, Fs, nperseg=NFFT, scaling = "spectrum")
        cout = np.sum( np.real(Pxyout[arg-1:arg+1]) )
        cout = cout/np.sum( np.sqrt(Nout[arg-1:arg+1]) )
        Cout.append(cout)

    Cin = np.array(Cin)/9.8e-9
    Cin = Cin*v_to_m_in*ac_corr

    Cout = np.array(Cout)/9.8e-9
    Cout = Cout*v_to_m_out*ac_corr
    
    return [Cin, Cout, [Cross, frequency[index0:index1]], Psd, Psdin, Psdout]


def gauss(x,a,b,c):
    g = c*np.exp(-0.5*((x-a)/b)**2)
    return g

def histo(c, bins):
    h,b = np.histogram(c, bins = bins)
    bc = np.diff(b)/2 + b[:-1]
    return [h, bc]

def linefit(x, a, b):
    return a*x + b


def gauss_filter(c1, c2, sigma, cross, psd, psdin, psdout):
    w1 = np.std(c1)
    w2 = np.std(c2)
    cnew1 = []
    cnew2 = []
    Cross = []
    Psd = []
    Psdin = []
    Psdout = []
    for i in range(len(c1)):
        if np.abs(c1[i]) <= np.abs(1.*sigma*w1) and np.abs(c2[i]) <= np.abs(1.*sigma*w2):
            cnew1.append(c1[i])
            cnew2.append(c2[i])
            Cross.append(cross[i])
            Psd.append(psd[i])
            Psdin.append(psdin[i])
            Psdout.append(psdout[i])
        else:
            print "excluded", i
    
    cnew1 = np.array(cnew1)
    cnew2 = np.array(cnew2)
    return [cnew1, cnew2, Cross, Psd, Psdin, Psdout]

cin, cout, cross, psd, psdin, psdout = corr(file_list_meas, v_to_m_in, v_to_m_out, folder_calibration, fres, True)
frequency = cross[1]
cross = cross[0]

bins = 30

for i in range(1):
    cin, cout, cross, psd, psdin, psdout = gauss_filter(cin, cout, 3, cross, psd, psdin, psdout)
#####

hin, bcin = histo(cin, bins)
hout, bcout = histo(cout, bins)

sigmain = np.sqrt(hin)
for i in range(len(sigmain)):
    if sigmain[i] == 0:
        sigmain[i] = 1.
sigmaout = np.sqrt(hout)
for i in range(len(sigmaout)):
    if sigmaout[i] == 0:
        sigmaout[i] = 1.

poptin, pcovin = opt.curve_fit(gauss, bcin, hin, sigma = sigmain)
poptout, pcovout = opt.curve_fit(gauss, bcout, hout, sigma = sigmaout)


vin = np.abs(poptin[1])**2
vout = np.abs(poptout[2])**2


v_combined = 1./(1./vin + 1./vout)

c = (1.*cin/vin + 1.*cout/vout)*v_combined

h, bc = histo(c, bins)
sigma = np.sqrt(h)
for i in range(len(sigma)):
    if sigma[i] == 0:
        sigma[i] = 1.
popt, pcov = opt.curve_fit(gauss, bc, h, sigma = sigma)


space = np.linspace(2*np.min(bc),2*np.max(bc), 1000)

acin = str(poptin[0]) + r"$\pm$" + str(np.sqrt(pcovin[0][0]))
acout = str(poptout[0]) + r"$\pm$" + str(np.sqrt(pcovout[0][0]))
ac = str(popt[0]) + r"$\pm$" + str(np.sqrt(pcov[0][0]))

print "acc in nano-g:"
print "accin =", acin
print "accout =", acout
print "acc =", ac

print poptin[1]
print poptout[1]
print popt[1]

colors = ['#1f78b4', '#e66101', '#33a02c', '#984ea3', '#F27781', '#18298C', '#04BF8A', '#F2CF1D', '#F29F05', '#7155D9', '#8D07F6', '#9E91F2', '#F29B9B', '#F25764', '#6FB7BF', '#B6ECF2', '#5D1314', '#B3640F']

plt.figure()
plt.plot(bcin, hin, "r.")
plt.plot(bcout, hout, "b.")
plt.plot(bc, h, "g.")
plt.plot(space, gauss(space, *poptin), "r-", label = "inloop")
plt.plot(space, gauss(space, *poptout), "b-", label = "outloop")
plt.plot(space, gauss(space, *popt), "g-", label = "combined")
plt.legend()
plt.tight_layout(pad = 0)

plt.figure(figsize=(5,3))
plt.rcParams.update({'font.size': 14})
plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = "o", color = colors[0])
plt.plot(space, gauss(space, *popt), "-", color = colors[1])
plt.xlabel("Acceleration [ng]")
plt.ylabel("Number of measurements")
plt.xlim(-30, 30)
plt.grid()
plt.tight_layout(pad = 0)

x = range(len(cin))
import matplotlib.cm as cm
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(x, cin, "r.", label = "inloop")
plt.subplot(3, 1, 2)
plt.plot(x, cout, "b.", label = "outloop")
plt.subplot(3, 1, 3)
plt.plot(x, c, "g.", label = "combined")
plt.tight_layout(pad = 0)

space = np.linspace(np.min(cin), np.max(cin), 1000)
poptL, pcovL = opt.curve_fit(linefit, cin, cout)

plt.figure()
plt.plot(space, space, "k--")
plt.scatter(cin, cout, c = x, cmap = cm.viridis)
plt.plot(space, linefit(space, *poptL), "r-", label = "fit")
plt.xlabel("inloop [ng]")
plt.ylabel("outloop [ng]")
plt.colorbar()
plt.legend()
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
plt.tight_layout(pad = 0)

npsd = np.zeros(len(psd[0]))
npsdin = np.zeros(len(psdin[0]))
npsdout = np.zeros(len(psdout[0]))

for j in psd:
    npsd = npsd + j
npsd = npsd/len(psd)

for k in psdin:
    npsdin = npsdin + k
npsdin = npsdin/len(psdin)

for j in psdout:
    npsdout = npsdout + j
npsdout = npsdout/len(psdout)

# space = np.linspace(np.min(bc)-5, np.max(bc)+5, 1000)

plt.figure()
plt.loglog(frequency, np.sqrt(npsd), label = "combined")
plt.loglog(frequency, np.sqrt(npsdin), label = "inloop")
plt.loglog(frequency, np.sqrt(npsdout), label = "outloop")
plt.xlabel("freq [Hz]")
plt.ylabel("ug/sqrt(Hz)")
plt.legend()
plt.xlim(1,100)
plt.tight_layout(pad = 0)

rho = 1800
diameter = 10.3e-6
def mass(Diameter, rho):
    m = (4/3.)*(np.pi)*((Diameter/2)**3)*rho
    return m

# n-g to atto-N (1e-18 N)
conversion_force = 1.*mass(diameter, rho)*(9.8e-9)*(1e18)
print "ng_to_aN conversion", conversion_force

plt.figure(figsize=(5,5))
plt.rcParams.update({'font.size': 14})
ax1 = plt.subplot(2,1,1)
ax1.semilogy(frequency, 1000.*np.sqrt(npsd))
ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("Sensitivity [ng$/\sqrt{Hz}$]")
ax1.set_xlim(1,100)
ax1.set_ylim(1e2,1e4)
ax2 = ax1.twinx()
mn, mx = ax1.get_ylim()
ax2.set_ylim(mn*conversion_force, mx*conversion_force) # this only changes the scale of the new y axis. Does not change the data.
ax2.set_ylabel('Sensitivity [aN$/\sqrt{Hz}$]')
ax2.set_yscale("log")
plt.subplot(2,1,2)
plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = "o", color = colors[0])
plt.plot(space, gauss(space, *popt), "-", color = colors[1])
plt.xlabel("Acceleration [ng]")
plt.ylabel("Number of measurements")
plt.xlim(-30, 30)
plt.grid()
plt.tight_layout(pad = 0)


plt.show()
