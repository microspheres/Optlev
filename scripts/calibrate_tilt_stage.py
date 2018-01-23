import numpy as np
import matplotlib
import matplotlib.pyplot as plt

filename = r'C:\Users\UsphereLab\Desktop\tilt_motor1.txt'

data = np.loadtxt(filename, skiprows=1)

print data

plt.figure()
plt.plot(data[:,0], data[:, 1:4], 'o')
plt.show()

af, bf = np.polyfit(data[:12,0],data[:12,2], 1)
ab, bb = np.polyfit(data[12:,0],data[12:,2], 1)

print af, ab, af/ab
