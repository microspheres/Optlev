import numpy as np
import h5py
import scipy.signal as sp
import matplotlib.pyplot as plt
import glob
import matplotlib.mlab as mlab
import scipy.optimize as opt
from scipy import signal

electron_charge = -1.6e-19 # coulomb
pi = np.pi

calibration1e = True
if not calibration1e :
    charge = -1*electron_charge # coulomb
else:
    charge = electron_charge

# important comment: The posititve X direction is choosen to be pointing away of the HV electrode and tongue. Empirically, for a sphere with
# negative charges (lots ot them), a positive DC voltage in the same electrode will make the sphere go towards negative X, therefore a negarive force
# which is nice because the force is Q*(-|e|) which is negative too, with Q = Capacitance*V > 0.
# For consistency, the calibration force with 1e must be negative and positive for a calibration with 1p.

mass = ((15/10)**3)*1e-12 # kg

NFFT = 2**16

HV_ch = 3
x_ch = 0

folder_cal = r"F:\data\20210601\15um_150umhole\2\calibration1e\fbx-0.1"

folder_meas = r"F:\data\20210601\15um_150umhole\2\no_field\fbx-0.1"

electrodes_distance = 0.0033

comsol_correction = 0.7

### files

filelist_calibration = glob.glob(folder_cal + "/*.h5")
filelist_meas = glob.glob(folder_meas + "/*.h5")

print(filelist_calibration)

def histogram(c, bins):
    h, b = np.histogram(c, bins = bins)
    bc = np.diff(b)/2. + b[:-1]
    return [h, bc]

def gauss(x, a, b, c):
    g = c * np.exp(-0.5 * ((x - a) / b) ** 2)
    return g

######## transfer functions

def harmonic_transfer_function(f, f0, gamma0, A):
    w = 2.*np.pi*f
    w0 = 2.*np.pi * f0
    g0 = 2.*np.pi*gamma0
    a = A/(w0**2 - w**2 + 1j*w*g0)
    return np.real(a)

def harmonic_psd(f, f0, gamma0, A):
    w = 2.*np.pi*f
    w0 = 2.*np.pi * f0
    g0 = 2.*np.pi*gamma0
    a = (A**2)/((w0**2 - w**2)**2 + (w*g0)**2)
    return a

def harmonic(f, f0, g, A):
    w = 2.*np.pi*f
    w0 = 2.*np.pi*f0
    G = 2.*np.pi*g
    a1 = 1.*np.sqrt(  (w**2 - w0**2)**2 + (w*G)**2  )
    return 1.*A/a1

#### CALIBRATION

def getdata(fname):
    print("Opening file: ", fname)
    f = h5py.File(fname, 'r')
    dset = f['beads/data/pos_data']
    dat = np.transpose(dset)
    Fs = dset.attrs['Fsamp']
    #time = dset.attrs["Time"]

    dat = dat * 10. / (2 ** 15 - 1)

    HV = dat[:, HV_ch] - np.mean(dat[:, HV_ch])
    x = dat[:, x_ch] - np.mean(dat[:, x_ch])

    # psd, f = mlab.psd(x, Fs=Fs, NFFT=NFFT)
    #
    # plt.figure()
    # plt.loglog(f, psd**0.5)
    # plt.show()

    return [Fs, x, HV]

def getnofield(nofieldlist):
    xpsd = 0
    for i in nofieldlist:
        data = getdata(i)
        Fs = data[0]
        x = data[1]
        psd, f = mlab.psd(x, Fs=Fs, NFFT=NFFT)
        xpsd = xpsd + psd
    xpsd = xpsd/len(nofieldlist)

    return [xpsd, f]

#[indexes, indexes2, psdhv, psdhv2]
def get_freqs_comb_calibration(list_calibration, plot):

    HVlist = []
    for i in list_calibration:
       data  = getdata(i)
       Fs = data[0]
       HVlist.append(data[2])

    psdhv = 0.
    psdhv2 = 0.
    for j in HVlist:
        # 1f peaks
        psd, f = mlab.psd(j, Fs=Fs, NFFT=NFFT)
        psdhv = psdhv + psd
        # 2f peaks (for induced dipole)
        m = j**2 - np.mean(j**2)
        psd2, f = mlab.psd(m, Fs=Fs, NFFT=NFFT)
        psdhv2 = psdhv2 + psd2

    psdhv = psdhv/len(HVlist)
    psdhv2 = psdhv2/len(HVlist)

    indexes = np.where(psdhv > 3e-6)[0]
    indexes2 = np.where(psdhv2 > 1e-11)[0]

    indexes2 = np.array(indexes2)

    if plot:
        plt.figure()
        plt.loglog(f, psdhv)
        plt.loglog(f[indexes], psdhv[indexes], "o")
        plt.loglog(f, psdhv2)
        plt.loglog(f[indexes2], psdhv2[indexes2], "s")

    #indexes, indexes2 = forbiden_freqs(f, indexes, indexes2, forbiden_list)

    # this is the voltage output of the synth into the electrodes
    voltage = np.sqrt(np.sum((f[1]-f[0])*psdhv[indexes]))
    print ("synth voltage RMS during electrode calibration in V = ", voltage)
    electrode_voltage = 200.*voltage

    electrode_voltage_1peak = 200.*np.sqrt(np.sum((f[1]-f[0])*psdhv[indexes[0:2]]))
    print("electrode voltage first peak RMS during electrode calibration in V = ", electrode_voltage_1peak)

    electrode_voltage = 200.*np.sqrt(np.sum((f[1]-f[0])*psdhv[indexes]))
    print("electrode voltage RMS during electrode calibration in V = ", electrode_voltage)

    if False:
        xpsd = 0
        plt.figure()
        for i in list_calibration:
            x = getdata(i)[1]
            f, psd = sp.csd(x, x, Fs, nperseg=NFFT, scaling="spectrum")
            xpsd = xpsd + psd/(np.sqrt(2.)*Fs/NFFT)
        xpsd = xpsd/len(list_calibration)
        plt.loglog(f, np.sqrt(xpsd), label = "1e")
        plt.loglog(f[indexes], np.sqrt(xpsd[indexes]), "s",)
        plt.xlabel("freq [Hz]")
        plt.ylabel("V/sqrt(Hz)")
        plt.legend()

    return [indexes, indexes2, psdhv, psdhv2]

#[fieldpsd, xpsdhv, f, index, df] in V^2 / Hz
def HV_field_and_X_psd(list_calibration, electrodes_distance, comsol_correction):
    data = get_freqs_comb_calibration(list_calibration, True)

    # field psd
    index = data[0]
    index2 = data[1]
    vpsd = data[2]
    vpsd = vpsd[index]
    fieldpsd = vpsd*(200.*comsol_correction/electrodes_distance)**2 ## remember that vpsd is in V^2/Hz that is why the ()**2

    # X psd in volts units (daq)
    Xlist = []
    for i in list_calibration:
       datax  = getdata(i)
       Fs = datax[0]
       Xlist.append(datax[1])

    xpsdhv = 0.
    for j in Xlist:
        psd, f = mlab.psd(j, Fs=Fs, NFFT=NFFT)
        xpsdhv = xpsdhv + psd

    df = f[1]-f[0]

    xpsdhv = xpsdhv/len(Xlist)

    xpsdhv = xpsdhv[index]
    f = f[index]

    return [fieldpsd, xpsdhv, f, index, df, index2]

def calibration_daqV_to_meters(list_calibration, electrodes_distance, comsol_correction):

    data = HV_field_and_X_psd(list_calibration, electrodes_distance, comsol_correction)
    fieldpsd = data[0]
    xpsdhv = data[1]
    f = data[2]
    df = data[4]
    index = data[3]
    index2 = data[5]

    p0 = [70., 1, 1]
    popt, pcov = opt.curve_fit(harmonic, f, xpsdhv**0.5, p0 = p0, sigma = xpsdhv**0.5)

    f0 = np.abs(popt[0])
    g = np.abs(popt[1])

    h = ((fieldpsd*df)**0.5)*(charge/mass)

    displacement_m = harmonic(f, f0, g, h)

    from_daq_to_m = np.mean( displacement_m / ((xpsdhv*df)**0.5) )

    from_daq_to_m = -np.abs(from_daq_to_m) # think about that, this number only depends on the optics and wiring of the daq, has nothing to do with charge.
    # the "issue" is that the charge changes sign above, but the xpsdhv is always positive regardless if there is 1e ou 1p at the calibration files.
    # important thing that sets the sign is the following, if the sphere has 1e and there is a positive voltage in the biased electrode, we know the force is
    # negative (coordinate going from biased electroded to grounded one). Does the choices makes this happen?

    if True:
        plt.figure()
        freqplot = np.linspace(30, 140, 1000)
        plt.loglog(f, np.abs(from_daq_to_m)*xpsdhv**0.5, "o")
        plt.loglog(freqplot, np.abs(from_daq_to_m)*harmonic(freqplot, *popt))
        plt.title("CHECK THIS FIT for the calibration")

    if False:
        #plt.figure()
        #plt.loglog(f, xpsdhv**0.5)
        #plt.loglog(f, harmonic(f, *popt))
        #plt.loglog(f, 1e7*displacement_m)
        #plt.figure()
        #plt.loglog(f, displacement_m/xpsdhv**0.5)
        #plt.loglog(f, from_daq_to_m*xpsdhv**0.5, ".")
        #plt.loglog(displacement_m, harmonic(f, *popt), ".")

        nofxpsd = getnofield(filelist_meas)

        #plt.figure()
        #plt.loglog(nofxpsd[1], from_daq_to_m*(nofxpsd[0]**0.5))
        #plt.figure()
        #plt.loglog(f, fieldpsd**0.5)

        force_s = mass*from_daq_to_m*((nofxpsd[0])**0.5)*np.sqrt(  ((2*np.pi*nofxpsd[1])**2 - (2*np.pi*f0)**2)**2 + (2*np.pi*nofxpsd[1]*2*np.pi*g)**2  )
        plt.figure()
        plt.loglog(nofxpsd[1], np.abs(force_s))
        #plt.show()

    #test force 1f with the calibration

    # testdata  = getdata(filelist_calibration[0])
    # Fs = testdata[0]
    # mcal = testdata[1]*from_daq_to_m
    # volt = testdata[2]
    #
    # f, c1 = sp.csd(volt, mcal, Fs, nperseg=NFFT, scaling = "spectrum")
    # f, c2 = sp.csd(volt, volt, Fs, nperseg=NFFT, scaling = "spectrum")
    #
    # fo = c1/(c2**0.5)
    # fo = fo*((2*np.pi)**2)*( f0**2 - f**2 + 1j*f*g  ) ## the -1j is due to the complex conjugate
    # fo = np.real(fo)
    # fo = mass*fo
    #
    # #print (np.sum(fo[index]**2)**0.5)
    #
    # sign = np.mean((fo[index]/np.abs(fo[index])))
    # print(sign)
    # print (sign*np.sum( (np.sum(fo[index] ** 2) ** 0.5) ))
    # print (charge*np.sum(fieldpsd*df)**0.5)
    #
    # plt.figure()
    # plt.loglog(f, np.abs(fo)) # the abs here is only because the force can be negative and this is a loglog plot. Not necessary otherwise
    # plt.plot(f[index], np.abs(fo[index]), ".")
    # plt.show()
    #quit()

    print (from_daq_to_m, f0, g)

    return [from_daq_to_m, index, index2, f0, g]

def force_psd(list_meas, list_calibration, electrodes_distance, comsol_correction):

    xpsd, f = getnofield(list_meas)

    from_daq_to_m, index, index2, f0, g = calibration_daqV_to_meters(list_calibration, electrodes_distance, comsol_correction)

    w0 = 2.*pi*f0
    G = 2.*pi*g
    w = 2.*pi*f

    h = (w**2 - w0**2)**2 + (w*G)**2

    fpsd = from_daq_to_m * mass * ((xpsd) ** 0.5) * np.sqrt(h)

    plt.figure()
    plt.loglog(f, np.abs(fpsd))

    return [f, np.abs(fpsd)]

def force_estimator_time(v_time_stream, v_to_m, f0, Gamma, Fs):

    # get fft of displacement

    displacement = v_time_stream*v_to_m

    time = np.linspace(0, (len(displacement))/Fs, len(displacement))

    xfft = scipy.fft.rfft(displacement)
    n = displacement.size
    freq = np.fft.rfftfreq(n, d = 1./Fs)

    #test to check the fft and inverse fft
    #plt.figure()
    #plt.plot(displacement)
    #new_dis = scipy.fft.irfft(xfft)
    #print (np.max(np.abs(np.imag(new_dis))))
    #print(np.min(np.abs(np.real(new_dis))))
    #plt.plot(new_dis)

    # do mass * 1/harmonic_transfer_function

    Hinv = 1./harmonic_transfer_function(freq, f0, Gamma, 1)

    force_fft = mass*Hinv*xfft

    # inverse fft

    force = scipy.fft.irfft(force_fft)

    # filter

    wn1 = 0.2*f0/(Fs/2)
    wn2 = 1.7*f0/(Fs/2)
    b, a = signal.butter(4, Wn = [wn1, wn2], btype = "bandpass")
    y = signal.filtfilt(b, a, force)

    bn, an = scipy.signal.iirnotch(60./(Fs/2), 80)
    bn2, an2 = scipy.signal.iirnotch(62.8 / (Fs / 2), 80)
    bn3, an3 = scipy.signal.iirnotch(49.4 / (Fs / 2), 80)
    bn4, an4 = scipy.signal.iirnotch(18.15 / (Fs / 2), 10)
    bn5, an5 = scipy.signal.iirnotch(29.35 / (Fs / 2), 20)
    bn6, an6 = scipy.signal.iirnotch(36.27 / (Fs / 2), 30)
    bn7, an7 = scipy.signal.iirnotch(40.63 / (Fs / 2), 30)
    bn8, an8 = scipy.signal.iirnotch(54.44 / (Fs / 2), 50)
    bn9, an9 = scipy.signal.iirnotch(72.6 / (Fs / 2), 80)
    bn10, an10 = scipy.signal.iirnotch(84.23 / (Fs / 2), 90)
    bn11, an11 = scipy.signal.iirnotch(80.95 / (Fs / 2), 90)
    bn12, an12 = scipy.signal.iirnotch(81.26 / (Fs / 2), 90)
    bn13, an13 = scipy.signal.iirnotch(180 / (Fs / 2), 100)
    bn14, an14 = scipy.signal.iirnotch(118.4 / (Fs / 2), 80)
    bn15, an15 = scipy.signal.iirnotch(120 / (Fs / 2), 80)

    y2 = signal.filtfilt(bn, an, y)
    y3 = signal.filtfilt(bn2, an2, y2)
    y3 = signal.filtfilt(bn3, an3, y3)
    y3 = signal.filtfilt(bn4, an4, y3)
    y3 = signal.filtfilt(bn5, an5, y3)
    y3 = signal.filtfilt(bn6, an6, y3)
    y3 = signal.filtfilt(bn7, an7, y3)
    y3 = signal.filtfilt(bn8, an8, y3)
    y3 = signal.filtfilt(bn9, an9, y3)
    y3 = signal.filtfilt(bn10, an10, y3)
    y3 = signal.filtfilt(bn11, an11, y3)
    y3 = signal.filtfilt(bn12, an12, y3)
    y3 = signal.filtfilt(bn13, an13, y3)
    y3 = signal.filtfilt(bn14, an14, y3)
    y3 = signal.filtfilt(bn15, an15, y3)

    plt.figure()
    plt.plot(time,y)
    #plt.plot(time,y2)
    plt.plot(time,y3)

    #print ( ((np.mean(y3[80000:120000]**2)**0.5)**2/(50))**0.5 )

    #test force psd
    psdforce, freqpsd = mlab.psd(force, Fs=Fs, NFFT=NFFT)
    psdfy, freqpsd = mlab.psd(y, Fs=Fs, NFFT=NFFT)
    psdfy3, freqpsd = mlab.psd(y3, Fs=Fs, NFFT=NFFT)
    plt.figure()
    plt.loglog(freqpsd, psdforce**0.5)
    plt.loglog(freqpsd, psdfy**0.5, label = "band pass only")
    plt.loglog(freqpsd, psdfy3 ** 0.5, label = "final")
    plt.legend()
    # print ( ((np.mean(y3[80000:120000]**2)**0.5)**2/(50))**0.5 )

    return [y3, time]


calibration_daqV_to_meters(filelist_calibration, electrodes_distance, comsol_correction)

force_psd(filelist_meas, filelist_calibration, electrodes_distance, comsol_correction)

plt.show()
