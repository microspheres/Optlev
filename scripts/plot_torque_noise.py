import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import glob


II = (8./15.)*np.pi*(1600.)*((7.5e-6)**5)

path_sphere = r"C:\data\20190725\15um_SiO2\4\pressures\daq"

path_keysight = r"C:\data\20190725\15um_SiO2\4\pressures\daq\KEYSIGHT_362337Hz"

file_sphere = glob.glob(path_sphere+"\*.npy")[0]

file_keysight = glob.glob(path_keysight+"\*.npy")[0]

savepath = path_sphere

datas = np.load(file_sphere)
datak = np.load(file_keysight)


plt.figure()
plt.loglog(datas[3], datas[3]*2.*np.pi*II*np.sqrt(datas[4]), label = "Sphere + DAQ")
plt.loglog(datak[3], datak[3]*2.*np.pi*II*np.sqrt(datak[4]), label = "Equivalent noise Keysight + DAQ")
plt.ylabel("Nm / $\sqrt{Hz}$")
plt.xlabel("Frequency [Hz]")
plt.grid()
plt.legend()
plt.tight_layout(pad = 0)
name = r"torque.pdf"
save = os.path.join(savepath, name)
plt.savefig(save)

plt.figure()
plt.loglog(datas[3], 2.*np.pi*np.sqrt(datas[4]), label = "Sphere + DAQ")
plt.loglog(datak[3], 2.*np.pi*np.sqrt(datak[4]), label = "Keysight + DAQ")
plt.ylabel("$\omega$ / $\sqrt{Hz}$")
plt.xlabel("Frequency [Hz]")
plt.grid()
plt.legend()
plt.tight_layout(pad = 0)
name = r"angular frequency.pdf"
save = os.path.join(savepath, name)
plt.savefig(save)

plt.figure()
plt.loglog(datas[3], datas[3]*2.*np.pi*np.sqrt(datas[4]), label = "Sphere + DAQ")
plt.loglog(datak[3], datak[3]*2.*np.pi*np.sqrt(datak[4]), label = "Equivalent noise Keysight + DAQ")
plt.ylabel("$\dot{\omega}$ / $\sqrt{Hz}$")
plt.xlabel("Frequency [Hz]")
plt.grid()
plt.legend()
plt.tight_layout(pad = 0)
name = r"angular acceleration.pdf"
save = os.path.join(savepath, name)
plt.savefig(save)

plt.figure()
plt.loglog(datas[3][1:], 2.*np.pi*np.sqrt(datas[4][1:])/datas[3][1:], label = "Sphere + DAQ")
plt.loglog(datak[3][1:], 2.*np.pi*np.sqrt(datak[4][1:])/datak[3][1:], label = "Keysight + DAQ")
plt.ylabel(r"$\theta$ / $\sqrt{Hz}$ [rad]")
plt.xlabel("Frequency [Hz]")
plt.grid()
plt.legend()
plt.tight_layout(pad = 0)
name = r"angle.pdf"
save = os.path.join(savepath, name)
plt.savefig(save)

plt.figure()
plt.errorbar(datas[2], datas[0], np.sqrt(datas[1]), fmt = ".-", label = "Sphere + DAQ")
plt.errorbar(datak[2], datak[0], np.sqrt(datak[1]), fmt = ".-", label = "Keysight + DAQ")
plt.ylabel(r"$\frac{\omega}{2 pi}$ [Hz]")
plt.xlabel("Time [s]")
plt.grid()
plt.legend()
plt.tight_layout(pad = 0)
name = r"frequency_vs_time.pdf"
save = os.path.join(savepath, name)
plt.savefig(save)

plt.show()
