import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import numpy, h5py, matplotlib

# https://www.mathworks.com/help/signal/ref/ellip.html
# https://en.wikipedia.org/wiki/Digital_filter

Fs = 100000 # fpga Fs

Ny = Fs/2

def transfer(tm, res, Fs):

    Fs = int(Fs)
    
    fft_c = np.fft.rfft(tm, tm.size)
    fft_r = np.fft.rfft(res, res.size)
    freq = np.fft.rfftfreq(tm.size, 1./Fs)

    t = np.abs(fft_r/fft_c)

    arg_c = 1.*np.angle(fft_c)
    arg_r = 1.*np.angle(fft_r)

    angle = arg_r - arg_c
    angle = ( angle + np.pi) % (2 * np.pi ) - np.pi

    
    return [freq, t, angle]

def filter_order_N(timestream, time, coef): # outputs the filtered timestream
    order = len(coef[0]) - 1
    B = coef[0]
    A = coef[1]

    A = np.array(A)
    B = np.array(B)

    Y = []

    for i in range(len(timestream)):

        Yaux = []
        Xaux = []
        for k in range(order):
            if i - k - 1 >= 0:
                Xaux.append(timestream[i - k - 1])
                Yaux.append(Y[i - k - 1])
            else:
                Yaux.append(0.)
                Xaux.append(0.)

        Yaux = np.array(Yaux)
        Xaux = np.array(Xaux)
        

        y = - np.sum(A[1:]*Yaux) + np.sum(B[1:]*Xaux) + B[0]*timestream[i]


        Y.append(y)

    return [Y, time]


f0 = 120. # Hz

Wn = 1.*f0/Ny

f02 = 60. # Hz

Wn2 = 1.*f02/Ny

order_list = [1,2,3,4]

time = np.linspace(0, 0.1, 10.*Fs)
timestream = np.random.randn(len(time))


plt.figure()
for i in order_list:


    coef = signal.ellip(i, 3., 30., Wn, btype = 'lowpass', output='ba')
    coef2 = signal.ellip(i, 3., 30., Wn2, btype = 'highpass', output='ba')
    
    resN, timeN = filter_order_N(timestream, time, coef)

    resN, timeN = filter_order_N(resN, time, coef2)

    freqN, tN, argN = transfer(np.array(timestream), np.array(resN), Fs)

    name = "order " + str(i)
    plt.subplot(2, 1, 1)
    plt.loglog(freqN, tN, label = name)
    plt.xlim(10, 1000)
    plt.ylim(1e-3, 2)
    plt.ylabel("|Transfer function|")
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.semilogx(freqN, argN, label = name)
    plt.xlim(10, 1000)
    plt.ylim(-np.pi, np.pi)
    plt.grid()
    plt.legend()
    plt.xlabel("Freq [Hz]")
    plt.ylabel("phase [rad]")
plt.show()
