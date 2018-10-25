import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

#### the configuration is: HWP, QWP, wolastron like prism followed by another wolastron like prism and a relative phase in between the paths, HWP and polarizer. 

def Interf(phase ,HWP1, QWP, HWP2): # gives the intensity after the polarizer.
    t = HWP1
    b = QWP
    g = HWP2

    i = 1.0*1j
    
    aux1 = 1.0*np.exp(i*phase)
    aux2 = 1.0*(np.cos(2*t)*( np.cos(b)**2 + i*np.sin(b)**2 ) + np.sin(2*t)*( (1.0-i)*np.sin(b)*np.cos(b) ))*np.cos(2*g)
    aux3 = 1.0*(np.sin(2*t)*( np.sin(b)**2 + i*np.cos(b)**2 ) + np.cos(2*t)*( (1.0-i)*np.sin(b)*np.cos(b) ))*np.sin(2*g)

    amp = aux1*aux2 + aux3
    intensity = amp*np.conjugate(amp)
    return intensity


phase = np.linspace(-pi, pi, 10000)


plt.figure()
plt.plot(phase, Interf(phase , pi/8, 0, pi/8), label = "HWP:QWP:HWP = pi/8:0:pi/8")
plt.plot(phase, Interf(phase , pi/2.65, pi/4, pi/8), label = "HWP:QWP:HWP = pi/2.65:pi/4:pi/8")
plt.plot(phase, Interf(phase , 0, pi/4, pi/8), label = "HWP:QWP:HWP = 0:pi/4:pi/8")
plt.legend()
plt.grid()
plt.show()

