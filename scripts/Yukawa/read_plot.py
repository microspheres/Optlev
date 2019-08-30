import numpy as np
import os
import matplotlib.pyplot as plt

LL = np.genfromtxt(os.path.join(r"C:\Users\yalem\OneDrive\Desktop\CL_droplets\Calculations", "master_new.txt"), delimiter = " ")


plt.figure()
plt.loglog(LL[:,0], LL[:,1], "k-")
plt.fill_between(LL[:,0], LL[:,1], 1.0e20*np.ones(len(LL[:,1])), alpha = 0.2)
plt.xlim(1.0e-6, 1.0e-4)
plt.ylim(10., 1.0e11)

plt.show()
