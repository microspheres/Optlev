import numpy as np
import matplotlib.mlab
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.signal as sp
import glob, os
import h5py
colors = ['#1f78b4', '#e66101', '#33a02c', '#984ea3', '#F27781', '#18298C', '#04BF8A', '#F2CF1D', '#F29F05', '#7155D9', '#8D07F6', '#9E91F2', '#F29B9B', '#F25764', '#6FB7BF', '#B6ECF2', '#5D1314', '#B3640F', ]

colors = colors + colors

Fs = 450.

dt = 1./Fs

NFFT = 2**10

pi = np.pi

M = 3.2e-12

kb = 1.38e-23

c = 1e-6/(1.) # from pixel to m

# Z-direction

fL1 = 80.
fL2 = 200.

fR1 = 42.
fR2 = 70.

f_avoid_L = [119., 121.]
p0R = [54., 20, 4e-9]
p0L = [124., 100., 4e-9]

fLmax = 170.


# X-direction

# day 11
#fL1 = 42.
#fL2 = 140.

# day 13
#fL1 = 65.
#fL2 = 140.

#fR1 = 20.
#fR2 = 50.

#f_avoid_L = [106., 113.]
#f_avoid_L = [69., 80.]
# p0R = [32., 70, 4]
# p0L = [74., 2., 4e-9]

#p0R = [37., 70, 4e-9]
#p0L = [53., 100., 4e-9]
#fLmax = 80.

pathlist = [r"C:\data\20200313\two_spheres\15um_german\1\1", r"C:\data\20200313\two_spheres\15um_german\1\2", r"C:\data\20200313\two_spheres\15um_german\1\3", r"C:\data\20200313\two_spheres\15um_german\1\4",]

#pathlist = [r"C:\data\20200313\two_spheres\15um_german\2\1", r"C:\data\20200313\two_spheres\15um_german\2\2", r"C:\data\20200313\two_spheres\15um_german\2\3", r"C:\data\20200313\two_spheres\15um_german\2\4", r"C:\data\20200313\two_spheres\15um_german\2\5", r"C:\data\20200313\two_spheres\15um_german\2\1_redo", r"C:\data\20200313\two_spheres\15um_german\2\2_redo", r"C:\data\20200313\two_spheres\15um_german\2\3_redo", r"C:\data\20200313\two_spheres\15um_german\2\4_redo", r"C:\data\20200313\two_spheres\15um_german\2\5_redo", ]

pathlist = [r"C:\data\20200311\two_spheres\15um_german\3\1", r"C:\data\20200311\two_spheres\15um_german\4\2", r"C:\data\20200311\two_spheres\15um_german\4\3", r"C:\data\20200311\two_spheres\15um_german\4\4", r"C:\data\20200311\two_spheres\15um_german\4\1_redo", r"C:\data\20200311\two_spheres\15um_german\4\2_redo", r"C:\data\20200311\two_spheres\15um_german\4\3_redo", r"C:\data\20200311\two_spheres\15um_german\4\4_redo",  ]

pathlist = glob.glob(os.path.join(r"C:\data\20200311\two_spheres\15um_german\4", "*", ""))
pathlist = pathlist[0:8]
#print pathlist

pathlist = [r"C:\data\20200313\two_spheres\15um_german\2\1", ]

savepsd = False

Use_Xdirection = False

#pathlist = [r"C:\data\20200311\two_spheres\15um_german\3\1"]

def get_pressANDpid(path):
    fname = glob.glob(path+"\*.h5")[0]
    f = h5py.File(fname,'r')
    dset = f['beads/data/pos_data']
    pid = dset.attrs['PID']
    Press = dset.attrs['pressures'][0]
    return [pid, Press]

def psd(f, f0, gamma, A):
    gamma = np.abs(gamma)
    a = A*gamma
    b = (f**2 - f0**2)**2 + (f*gamma)**2

    return a/b

def getinfo_X(pathlist):
    TempL = []
    TempR = []
    DampL = []
    DampR = []
    Dgx = []
    plt.figure()
    counter = 0
    for i in pathlist:
        dgx = get_pressANDpid(i)[0][0]
        Dgx.append(dgx)
        filename = glob.glob(i+"\*.npy")[0]
        Data1L, Data1R, Data1Lx, Data1Rx = np.load(filename, encoding = 'latin1')

        if Use_Xdirection:
            DataL = c*Data1Lx
            DataR = c*Data1Rx
        else:
            DataL = c*Data1L
            DataR = c*Data1R
            

        psd1L, freqs = matplotlib.mlab.psd(DataL, NFFT = NFFT, Fs = Fs)
        psd1R, freqs = matplotlib.mlab.psd(DataR, NFFT = NFFT, Fs = Fs)

        df = freqs[1] - freqs[0]

        iL0 = np.where(freqs > fL1)[0][0]
        iL1 = np.where(freqs > fL2)[0][0]

        fit_points_L0 = np.logical_and(freqs > fL1, freqs < f_avoid_L[0])
        fit_points_L1 = np.logical_and(freqs > f_avoid_L[1], freqs < fL2)
        fit_L = fit_points_L0 + fit_points_L1
        

        try:
            poptL, pcovL = curve_fit(psd, freqs[fit_L], psd1L[fit_L], p0 = p0L,  ) #sigma = np.sqrt(psd1L[iL0:iL1]) )
            fitL = True
        except:
            fitL = False
            print "fitL failed"
            poptL = p0L
        fa = np.linspace(1, 200, 2000)

        iR0 = np.where(freqs > fR1)[0][0]
        iR1 = np.where(freqs > fR2)[0][0]

        poptR, pcovR = curve_fit(psd, freqs[iR0:iR1], psd1R[iR0:iR1], p0 = p0R,)#sigma = np.sqrt(psd1R[iR0:iR1]) )
        #poptR = p0R

        fb = np.linspace(1, 200, 2000)

        if fitL:
            print poptL
            TL = (df*np.sum(psd(freqs, *poptL))*(2*pi*poptL[0])**2)*M/(pi*kb)
            TR = (df*np.sum(psd(freqs, *poptR))*(2*pi*poptR[0])**2)*M/(pi*kb)
            if poptL[0] > fLmax:
                TL = (df*np.sum(psd(freqs, *poptL))*(2*pi*fLmax)**2)*M/(pi*kb)
        else:
            TR = (df*np.sum(psd(freqs, *poptR))*(2*pi*poptR[0])**2)*M/(pi*kb)
            TL = 0
            

        TempL.append(TL)
        TempR.append(TR)
        DampL.append(2.*pi*np.abs(poptL[1]))
        DampR.append(2.*pi*np.abs(poptR[1]))

        LabelL = "Bottom with Feedback,", " f0 = " + str("%.1f" % poptL[0])
        LabelR = "Top", " f0 = " + str("%.1f" % poptR[0])


        f, Pxy = np.abs( np.real( sp.csd(DataL, DataR, Fs, nperseg=NFFT, scaling = "spectrum") ) )
        f, Pxx = sp.csd(DataL, DataL, Fs, nperseg=NFFT, scaling = "spectrum")
        f, Pyy = sp.csd(DataR, DataR, Fs, nperseg=NFFT, scaling = "spectrum")
        Cxy = (Pxy**2)/(Pxx*Pyy)

        ColorL = colors[counter]
        ColorR = colors[counter+2]
        #plt.loglog(freqs[fit_L], psd1L[fit_L], label = LabelL, color = ColorL)
        #plt.loglog(freqs[iR0:iR1], psd1R[iR0:iR1], label = LabelR, color = ColorR)
        plt.loglog(freqs, psd1L, label = LabelL, color = ColorL)
        plt.loglog(freqs, psd1R, label = LabelR, color = ColorR)

        plt.loglog(fa, psd(fa, *poptL), color = ColorL)
        plt.loglog(fb, psd(fb, *poptR), color = ColorR)

        plt.xlabel("Frequency [Hz]")
        plt.ylabel("$S_{zz}$ [m$^2$/Hz]")

        # plt.legend()
        # plt.yscale("linear")
        plt.tight_layout()
        if savepsd:
            plt.savefig(i+r'\PSD.pdf')
    

        # plt.figure()
        # plt.semilogx(f, Cxy)
        # plt.xlabel("Frequency [Hz]")
        # plt.ylabel("Real of Normalized CSD")
        # plt.tight_layout()
        # plt.savefig('CSD.pdf')
        counter = counter + 1

    return [TempL, TempR, DampL, DampR, Dgx]

TempL, TempR, DampL, DampR, Dgx = getinfo_X(pathlist)

plt.figure()
#plt.plot(Dgx, TempL, "o", label = "Bottom temp")
plt.plot(Dgx, TempR, "o", label = "Top temp")
plt.xlabel("xDerivative Gain")
plt.legend()

plt.figure()
#plt.plot(Dgx, DampL, "o", label = "Bottom damp")
plt.plot(Dgx, DampR, "o", label = "Top damp")
plt.xlabel("xDerivative Gain")
plt.legend()

plt.show()
