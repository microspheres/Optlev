import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy.optimize as opt


# format is [freqhp, freq_meas, xpsd2_m_hp, xpsd2_m_meas, xpsd2_v_hp, xpsd2_meas_v]

loadfolder = r"C:\data\20191014\22um\prechamber_LP\5\meas"
loadfile = glob.glob(loadfolder+"\*.npy")[0]

loadfolder2 = r"C:\data\20191014\22um\prechamber_ATM\3\calibration2p"
loadfile2 = glob.glob(loadfolder2+"\*.npy")[0]

loadfolder3 = r"C:\data\20190812\22um_SiO2\4\calibration1e"
loadfile3 = glob.glob(loadfolder3+"\*.npy")[0]

loadfolder4 = r"C:\data\20190812\22um_SiO2\2\calibration1_p"
loadfile4 = glob.glob(loadfolder4+"\*.npy")[0]

data = np.load(loadfile)
data2 = np.load(loadfile2)
data3 = np.load(loadfile3)
data4 = np.load(loadfile4)

plt.figure()
plt.loglog(data[0], data[4])
plt.loglog(data[1], data[5], label = "Pre chamber < 1mbar")

plt.loglog(data2[0], data2[4])
plt.loglog(data2[1], data2[5], label = "Pre chamber 1 ATM")

plt.loglog(data3[0], data3[4])
plt.loglog(data3[1], data3[5], label = "No Pre chamber")

plt.loglog(data4[0], data4[4])
plt.loglog(data4[1], data4[5], label = "No Pre chamber")
plt.xlim(1, 150)
plt.ylim(2e-8, 1e-1)
plt.xlabel("freq [Hz]")
plt.ylabel("PSD V^2/Hz")
plt.legend()
plt.grid()


def psd(f, A, f0, gamma):
    w0 = 2.*np.pi*f0
    w = 2.*np.pi*f
    gamma1 = 2.0*np.pi*gamma
    s1 = 2.*A*(gamma1*(w0**2))
    s2 = 1.*(w0**2)*((w0**2 - w**2)**2 + (gamma1*w)**2)
    s = np.sqrt(s1/s2)
    return A*s


def normalization_voltage(filename):
    a = np.load(filename)
    freqHP = a[0]
    xpsdHP2 = a[4]
    popt, pcov = opt.curve_fit(psd, freqHP[20:1000], np.sqrt(xpsdHP2[20:1000]), p0 = [400, 85, 50])
    # print popt
    # plt.figure()
    # plt.loglog(freqHP[20:1000], np.sqrt(xpsdHP2[20:1000]))
    # plt.loglog(freqHP, psd(freqHP, *popt))
    # plt.show()

    k = psd(freqHP, *popt)[0]**2
    plt.figure()
    label = str(filename)
    plt.loglog(freqHP, xpsdHP2/k, label = label)
    plt.loglog(freqHP, psd(freqHP, *popt)**2/k)
    plt.loglog(a[1], a[5]/k)
    plt.xlim(1, 150)
    plt.legend()

    

    
# normalization_voltage(loadfile3)
# loadfiles = [loadfile, loadfile2, loadfile3, loadfile4]
# for i in loadfiles:
#     normalization_voltage(i)
# plt.grid()
plt.show()
