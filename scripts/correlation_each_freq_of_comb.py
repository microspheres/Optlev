import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import os, re, time, glob
from scipy.signal import butter, lfilter, filtfilt
from scipy import signal
import bead_util as bu
from scipy.optimize import curve_fit
import glob

NFFT = 2**18
Fs = 10000.

bins = 60 # hist bins

order = 2 # order of the filter. DO NOT CHANGE!

p = bu.drive
p1 = bu.xi

choose_data_index = True
choose_files = False

#############################
#############################
#############################
#starting the code: check below!


steps = "False"
vth = 0. # is the steps threshold. Most of the cases it is zero.
transfer = "False" # enable the transfer function, does not use correlation
calibration_mode = "False"
force_acc = "False" # to be used with a measurement with no ac field
use_noise = False

mass = (2.58*10**-12) # in kg

Number_of_e = (7.76*10**14)

distance = 0.0021 # meters

V_calibration = 1.0 #as shown on the daq
V_meas_ac = 20.0 # as shown on the daq
V_max_ac = 20.0 # used on the force and acceleration... as shown on daq


freq_list = [48.]

path_charge = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\several_distances\back_and_M2tilt\1\calibration1p"

path_signal = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\several_distances\back_and_M2tilt\1\meas1"

path_noise = r"C:\data\20180323\bead3_SiO2_15um_POL_NS\several_distances\back_and_M2tilt\1\meas1"

endfile = -1

startfile = 0

start_index = int(2**18 - 1)
end_index = -1

file_list_signal = glob.glob(path_signal+"\*.h5")
file_list_charge = glob.glob(path_charge+"\*.h5")
file_list_noise = glob.glob(path_noise+"\*.h5")

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

file_list_signal = list_file_time_order(file_list_signal)
if choose_files:
    file_list_signal = file_list_signal[startfile:endfile]

################################################ sensitivity
if force_acc == "True":
    V_meas_ac = V_max_ac


################################################ calibration mode
if calibration_mode == "True":
    V_meas_ac = V_calibration
    file_list_signal = file_list_charge
    steps = "False"
    Number_of_e = 1.0


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


def getdata_x_d(fname):
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		Fs = dset.attrs['Fsamp']
		dat = dat * 10./(2**15 - 1)
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )
        
        if choose_data_index:
            x = dat[start_index:end_index, p1]-numpy.mean(dat[start_index:end_index, p1])
            drive = dat[start_index:end_index, p] - numpy.mean(dat[start_index:end_index, p])
        else:
            x = dat[:, p1]-numpy.mean(dat[:, p1])
            drive = dat[:, p] - numpy.mean(dat[:, p])
        
        return [x, drive]



def drive(file_list_charge, ind): # returns nice drive or x depending on the index
    d = 0
    for i in range(len(file_list_charge)):
        aux = getdata_x_d(file_list_charge[i])[ind]
        d = d + aux
        print "wait"
    d = d/np.max(d)
    return d


def arg_each_freq(d, freq_list): # returns argument
    A = [] #list of fft of drive at each freq of the list
    ARG = [] # list of the arguments
    for i in range(len(freq_list)):
        daux = butter_bandpass_filter(d, freq_list[i]-1, freq_list[i]+1, Fs, order)
        dw = np.fft.rfft(daux)
        arg = np.argmax(np.abs(dw))
        # A.append(dw)
        ARG.append(arg)
    return ARG


def xtemplate_charge(file_list_charge, arg): # returns fft of xcharge at the freqs
    fft = 0
    for i in range(len(file_list_charge)):
        x = getdata_x_d(file_list_charge[i])[0]
        aux = np.fft.rfft(x*np.hanning(len(x)))
        fft = fft + aux
    xt = []
    fft = fft/len(file_list_charge)
    for j in range(len(arg)):
        aux2 = fft[arg[j]]
        xt.append(aux2)           
    return xt


def jnoise(file_list_noise,arg): # returns noise psd at each freq of the list
    psd = 0
    for i in range(len(file_list_charge)):
        x = getdata_x_d(file_list_charge[i])[0]
        aux = np.fft.rfft(x)
        aux = np.abs(np.conjugate(aux)*aux)
        psd = psd + aux
    jpsd_arg = []
    for j in range(len(arg)):
        aux2 = psd[arg[j]]**0
        if use_noise:
            aux2 = psd[arg[j]]
        jpsd_arg.append(aux2)           
    return jpsd_arg


def corr(xt, list_signal, jpsd_arg, arg): #return corr[file][freq]
    corr = []
    for i in range(len(list_signal)):
        x = getdata_x_d(list_signal[i])[0]
        xfft = np.fft.rfft(x*np.hanning(len(x)))
        clist = []
        print "doing", 1.*i/len(list_signal)
        for j in range(len(arg)):
            xaux = xfft[arg[j]]
            caux = np.conjugate(xt[j])*xaux/jpsd_arg[j]
            clist.append(caux)
        corr.append(clist)
    return corr

def corr_allfreq(corr): #return corr considering all freq
    c = []
    for i in range(len(corr)):
        a = np.sum(corr[i])
        c.append(a)
    return c

def auto_calibration(xt, jpsd_arg): # calibrates automatic
    c = []
    for i in range(len(xt)):
        aux = np.conjugate(xt[i])*xt[i]/jpsd_arg[i]
        c.append(aux)
    s = np.sum(c)
    cali = 1.0/s
    cali = cali*(V_calibration/V_meas_ac)/(Number_of_e)
    return cali

def organize_DC_pos(list):
    file_list_new = []
    for i in range(len(list)):
        if float(re.findall("-?\d+mVdc",list[i])[0][:-4]) > vth:
            file_list_new.append(list[i])
    return file_list_new

def organize_DC_neg(list):
    file_list_new = []
    for i in range(len(list)):
        if float(re.findall("-?\d+mVdc",list[i])[0][:-4]) < -vth:
            file_list_new.append(list[i])
    return file_list_new

########################################################################
########################################################################
# steps true or false? see below!
########################################################################
########################################################################


#things below are not to change!
d = drive(file_list_charge, 1)
arg = arg_each_freq(d, freq_list)
xt = xtemplate_charge(file_list_charge,arg)
jpsd_arg = jnoise(file_list_noise, arg)
cali = auto_calibration(xt, jpsd_arg)
#things up are not to change!

################## if steps are false!

if steps == "False":
    corr = corr(xt, file_list_signal, jpsd_arg, arg)
    corrfreq = corr_allfreq(corr)

    print "mean correlation in e number"
    print np.real(np.mean(corrfreq)*cali)
    
    if calibration_mode == "False":
        corrfreq = np.array(corrfreq)
        plt.figure()
        plt.plot(np.real(corrfreq*cali), "ro")
        plt.grid()

        ############################# for histogram
        
        s = np.real(corrfreq*cali)

        def gauss(x,a,b,c):
            g = c*np.exp(-0.5*((x-a)/b)**2)
            return g

        h,b = np.histogram(s, bins = bins)

        bc = np.diff(b)/2 + b[:-1]

        p0 = [np.mean(s), np.std(s)/np.sqrt(len(s)), 30]
        try:
            popt, pcov = curve_fit(gauss, bc, h, p0)
        except:
            popt = p0
            pcov = np.zeros([len(p0),len(p0)])


        space = np.linspace(bc[0],bc[-1], 1000)

        label_plot = str(popt[0]) + " $\pm$ " + str(np.sqrt(pcov[0,0]))

        print "result from charge fit in e#"
        print popt
        print np.sqrt(pcov[0,0])

        plt.figure()
        plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = 'ko', label = label_plot)
        plt.plot(space, gauss(space,*popt))
        plt.xlabel("Electron Number")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(path_signal,'histogram.pdf'))
        if transfer == "False" and force_acc == "False":
            plt.show()
            
        if force_acc == "True": ### force and acceleration
            
            # force
            
            e_charge = 1.6e-19 # SI
            cali = cali*(V_max_ac/V_calibration)*Number_of_e
            force = np.real(corrfreq*cali)*e_charge*(200.*0.5*V_calibration/distance)
            
            h,b = np.histogram(force, bins = bins)

            bc = np.diff(b)/2 + b[:-1]

            p0 = [4.*10**-19, 10.*10**-19, 2]
            try:
                popt, pcov = curve_fit(gauss, bc, h, p0)
            except:
                popt = p0

            space = np.linspace(bc[0],bc[-1], 1000)

            print "result from force fit in N"
            print popt
            print np.sqrt(pcov[0,0])

            label_plot = str(popt)

            plt.figure()
            plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = 'ko', label = label_plot)
            plt.plot(space, gauss(space,*popt))
            plt.xlabel("Force [N]")
            plt.grid()
            
            ## acceleration
            
            acc = force/mass
            
            h,b = np.histogram(acc, bins = bins)

            bc = np.diff(b)/2 + b[:-1]

            p0 = [4.*10**-7, 5.*10**-6, 2]
            try:
                popt, pcov = curve_fit(gauss, bc, h, p0)
            except:
                popt = p0

            space = np.linspace(bc[0],bc[-1], 1000)

            print "result from acc fit in m/s^2"
            print popt
            print np.sqrt(pcov[0,0])

            plt.figure()
            plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = 'ko')
            plt.plot(space, gauss(space,*popt))
            plt.xlabel("Acceleration [m/$s^2$]")
            plt.ylabel("Number of measurements")
            plt.grid()
            print "number of files", len(file_list_signal)
            plt.show()
            # np.savetxt(os.path.join(r"C:\data\acceleration_paper\from_dates\20171004bead8_23um_QWP_NS",'hist_ms2.txt'), (bc,h))
            if transfer == "False":
                plt.show()
                

###################
################# if steps are true!


if steps == "True":

    file_list_pos = organize_DC_pos(list_file_time_order(file_list_signal))
    file_list_neg = organize_DC_neg(list_file_time_order(file_list_signal))

    l1 = len(file_list_pos)
    l2 = len(file_list_neg)
    lmin = np.min([l1,l2])

    file_list_pos = file_list_pos[:lmin]
    file_list_neg = file_list_neg[:lmin]

    corrpos = corr(xt, file_list_pos, jpsd_arg, arg)
    corrfreqpos = np.array(corr_allfreq(corrpos))
    corrneg = corr(xt, file_list_neg, jpsd_arg, arg)
    corrfreqneg = np.array(corr_allfreq(corrneg))

    corrfreq_total = 0.5*(corrfreqneg + corrfreqpos)

    print "mean correlation in e number from steps"
    print np.real(np.mean(corrfreq_total)*cali)
    
    plt.figure()
    plt.plot(np.real(corrfreq_total*cali), "ko")
    plt.plot(np.real(corrfreqneg*cali), "ro")
    plt.plot(np.real(corrfreqpos*cali), "go")
    plt.grid()

    ################# for histogram
    
    s = np.real(corrfreq_total*cali)

    def gauss(x,a,b,c):
        g = c*np.exp(-0.5*((x-a)/b)**2)
        return g

    h,b = np.histogram(s, bins = bins)

    bc = np.diff(b)/2 + b[:-1]

    p0 = [np.mean(s), np.std(s)/np.sqrt(len(s)), 15]
    try:
        popt, pcov = curve_fit(gauss, bc, h, p0)
    except:
        popt = p0
        pcov = np.zeros([len(p0),len(p0)])
    space = np.linspace(bc[0],bc[-1], 1000)

    print "result from charge fit in e#"
    print popt
    print np.sqrt(pcov[0,0])

    label_plot = str(popt[0]) + " $\pm$ " + str(np.sqrt(pcov[0,0]))

    plt.figure()
    plt.errorbar(bc, h, yerr = np.sqrt(h), fmt = 'ko')
    plt.plot(space, gauss(space,*popt), label = label_plot)
    plt.legend()
    plt.xlabel("Electron Number")
    plt.grid()
    plt.savefig(os.path.join(path_signal,'histogram.pdf'))
    if transfer == "False":
        plt.show()



##############################################
##############################################
##############################################

# transfer function

path_save = path_signal

if transfer == "True":
    d = drive(file_list_charge,1)
    dfft = np.fft.rfft(d)
    x = drive(file_list_signal,0)
    x = x*np.hanning(len(x))
    xfft = np.fft.rfft(x)
    AR = arg_each_freq(d, freq_list)
    tf = drive(file_list_charge,0)
    tf = np.fft.rfft(tf)
    freq = np.fft.rfftfreq(len(d), 1./Fs)

    # print AR

    def plot(dfft, tf = []):
        plt.figure()
        plt.loglog(freq, np.abs(dfft))
        A = np.zeros(len(AR))
        for i,a in enumerate(AR):
            plt.plot(freq[a], np.abs(dfft[a]), "o")
            A[i] = np.abs(dfft[a])
            if len(tf) > 0:
                s = np.median(A/np.abs(tf[AR]))
                plt.plot(freq[AR],np.abs(tf[AR]*s), "r")

    plot(dfft)
    plot(xfft, tf)
    plt.xlim(5, 3500)
    plt.ylim(1e-2, 1e5)
    plt.xlabel("Freq [Hz]")
    plt.ylabel("Transfer function with no charge")
    plt.grid()
    plt.savefig(os.path.join(path_save,'Transfer_no_charge.pdf'))

################################### transfer with charge: For no charge see above
    d = drive(file_list_charge,1)
    dfft = np.fft.rfft(d)
    x = drive(file_list_charge,0)
    xfft = np.fft.rfft(x)
    AR = arg_each_freq(d, freq_list)
    tf = drive(file_list_charge,0)
    tf = np.fft.rfft(tf)
    freq = np.fft.rfftfreq(len(d), 1./Fs)

    # print AR

    def plot(dfft, tf = []):
        plt.figure()
        plt.loglog(freq, np.abs(dfft))
        A = np.zeros(len(AR))
        for i,a in enumerate(AR):
            plt.plot(freq[a], np.abs(dfft[a]), "o")
            A[i] = np.abs(dfft[a])
            if len(tf) > 0:
                s = np.median(A/np.abs(tf[AR]))
                plt.plot(freq[AR],np.abs(tf[AR]*s), "r")

    plot(dfft)
    plot(xfft, tf)
    plt.xlim(5, 3500)
    plt.ylim(1e-2, 1e5)
    plt.xlabel("Freq [Hz]")
    plt.ylabel("Transfer function with 1 charge")
    plt.grid()
    plt.savefig(os.path.join(path_save,'Transfer_with_1charge.pdf'))

    plt.show()

