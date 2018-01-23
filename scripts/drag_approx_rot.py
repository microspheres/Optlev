import numpy, h5py, matplotlib
import numpy as np

eta = 18.*10**-6

p = 1.*10**-5 # Pa

R = 3.*10**-6

M = (1.18*10**-11)/(3.6)

m = 4.7*10**-26

kb = 1.38*10**-23

kn = (1/R)*(eta/p)*np.sqrt(np.pi*kb*300/(2*m))

ck = 0.31*kn/(0.785 + 1.152*kn + kn**2)

Gamma0 = (6.*np.pi*eta*R/M)*(0.619/(0.619 + kn))*(1 + ck)

print Gamma0
