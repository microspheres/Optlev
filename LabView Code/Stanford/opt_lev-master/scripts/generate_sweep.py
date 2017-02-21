import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


t = np.linspace(0, 1, 16000)
wfm = scipy.signal.chirp(t, f0 = 10, f1 = 200, t1 = 1, method = 'linear', phi = 90)

np.savetxt('awesome.wfm', wfm)

plt.plot(t, wfm)



plt.show()

