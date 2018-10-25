import numpy, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

matplotlib.rcParams.update({'font.size': 16})
hspace = 0.08
ylabel = "$f_{rot}$ [MHz]"

## way of saving
#np.savetxt('data.txt', (Tdata, rotdata, pressdata, res))
#np.savetxt('fit.txt', (Tfit, rot_fit))

path_vaterite1 = r"C:\data\201712018\bead2_um_QWP_NS_VAT"

file_vaterite1_data = glob.glob(path_vaterite1+"\*data.txt")
file_vaterite1_fit = glob.glob(path_vaterite1+"\*fit.txt")

data_vaterite1 = np.loadtxt(file_vaterite1_data[0])
fit_vaterite1 = np.loadtxt(file_vaterite1_fit[0])



path_SiO2_1 = r"C:\data\20180129\bead3_um_POL_NS_SiO2_10um\meas18_V_-8.5_continuous"

file_SiO2_1_data = glob.glob(path_SiO2_1+"\*data.txt")
file_SiO2_1_fit = glob.glob(path_SiO2_1+"\*fit.txt")

data_SiO2_1 = np.loadtxt(file_SiO2_1_data[0])
fit_SiO2_1 = np.loadtxt(file_SiO2_1_fit[0])



path_SiO2_2 = r"C:\data\20180129\bead1_um_POL_NS_SiO2_10um\meas_8_V-2.5"

## way of saving for sphere SiO2_2
#np.savetxt('data.txt', (Tdata, rotdata, pressdata, res))
#np.savetxt('fit.txt', (Tfit, rot_fit, rot_fit2))

file_SiO2_2_data = glob.glob(path_SiO2_2+"\*data.txt")
file_SiO2_2_fit = glob.glob(path_SiO2_2+"\*fit.txt")

data_SiO2_2 = np.loadtxt(file_SiO2_2_data[0])
fit_SiO2_2 = np.loadtxt(file_SiO2_2_fit[0])

################################################

# fig1 = plt.figure()
# g = gs.GridSpec(2,1, hspace = hspace)
# plt.subplot(g[0])
# im =plt.scatter(data_vaterite1[0], data_vaterite1[1]/1e6, s=7, c=data_vaterite1[2]*1e7)
# plt.plot(fit_vaterite1[0], fit_vaterite1[1]/1e6, "r--", lw = 1.5)

# plt.grid()
# plt.xlim(-100, 1200)
# plt.ylim(2.5, 6)
# plt.ylabel(ylabel)
# plt.xlabel("Time [s]")
# plt.legend(loc="lower right", frameon = False)
# plt.gca().set_xticklabels([])

# plt.subplot(g[1])
# plt.scatter(data_vaterite1[0], data_vaterite1[3]*100, s=7, c=data_vaterite1[2]*1e5)

# plt.ylim(-0.3, 0.3)
# plt.xlim(-100, 1200)
# plt.ylabel("Residuals [%]")
# plt.xlabel("Time [s]")
# plt.grid()
# plt.legend(loc="upper right", frameon = False)
# plt.legend()
# plt.gcf().set_size_inches(6.4,5)
# fig1.subplots_adjust(right=0.8)
# cbar_ax = fig1.add_axes([0.81, 0.1, 0.03, 0.88])
# fig1.colorbar(im, cax=cbar_ax, label = r"Pressure [$\times 10^{-7}$mbar]")
# plt.subplots_adjust(right = 0.80, top = 0.98, left = 0.16, bottom = 0.13)
# #plt.show()

# #######################################################

# fig2 = plt.figure()
# g = gs.GridSpec(2,1, hspace = hspace)
# plt.subplot(g[0])
# im = plt.scatter(data_SiO2_1[0], data_SiO2_1[1]/1e6, s=7, c=data_SiO2_1[2]*1e7)
# plt.plot(fit_SiO2_1[0], fit_SiO2_1[1]/1e6, "r--", lw = 1.5)

# plt.grid()
# plt.ylim(0.5,5)
# plt.xlim(-3000,105000)
# plt.ylabel(ylabel)
# plt.xlabel("Time [s]")
# plt.legend(loc="lower right", frameon = False)
# plt.gca().set_xticklabels([])
# plt.xticks([0.0, 30000, 60000, 90000])


# plt.subplot(g[1])
# plt.scatter(data_SiO2_1[0], data_SiO2_1[3]*100, s=7, c=data_SiO2_1[2]*1e7)

# plt.ylim(-1.2,1.2)
# plt.xlim(-3000,105000)
# plt.ylabel("Residuals [%]")
# plt.xlabel("Time [s]")

# plt.grid()
# plt.legend(loc="upper right", frameon = False)
# plt.xticks([0.0, 30000, 60000, 90000])
# plt.legend()
# plt.gcf().set_size_inches(6.4,5)
# fig2.subplots_adjust(right=0.8)
# cbar_ax = fig2.add_axes([0.81, 0.1, 0.03, 0.88])
# fig2.colorbar(im, cax=cbar_ax, label = r"Pressure[$\times 10^{-7}$mbar]")
# plt.subplots_adjust(right = 0.750, top = 0.98, left = 0.12, bottom = 0.13)
# #plt.show()

# ################################################################

# fig3 = plt.figure()
# g = gs.GridSpec(2,1, hspace = hspace)
# plt.subplot(g[0])
# im = plt.scatter(data_SiO2_2[0], data_SiO2_2[1]/1e6, s=7, c=data_SiO2_2[2]*1e7,  vmin=1.67, vmax=1.96)
# plt.plot(fit_SiO2_2[0], fit_SiO2_2[1]/1e6, "r--", lw = 1.5)
# plt.plot(fit_SiO2_2[0], fit_SiO2_2[2]/1e6, "r--", lw = 1.5)

# plt.grid()
# plt.ylim(0.3,0.850)
# plt.xlim(-3000, 90000)
# plt.ylabel(ylabel)
# plt.xlabel("Time [s]")
# plt.legend(loc="lower right", frameon = False)
# plt.xticks([0.0, 40000., 80000.])
# plt.gca().set_xticklabels([])

# plt.subplot(g[1])
# plt.scatter(data_SiO2_2[0], data_SiO2_2[3]*100, s=7, c=data_SiO2_2[2]*1e7,  vmin=1.67, vmax=1.96)

# plt.ylim(-0.45, 0.45)
# plt.xlim(-3000, 90000)
# plt.ylabel("Residuals [%]")
# plt.xlabel("Time [s]")
# plt.grid()
# plt.legend(loc="upper right", frameon = False)
# plt.xticks([0.0, 40000., 80000.])
# plt.legend()
# plt.gcf().set_size_inches(6.4,5)
# fig3.subplots_adjust(right=0.8)
# cbar_ax = fig3.add_axes([0.81, 0.1, 0.03, 0.88])
# fig3.colorbar(im, cax=cbar_ax, label = r"Pressure[$\times 10^{-7}$mbar]")
# plt.subplots_adjust(right = 0.80, top = 0.98, left = 0.185, bottom = 0.13)

# plt.show()









from matplotlib.font_manager import FontProperties
from matplotlib import rc

##### crazy subplot

fig1 = plt.figure()
g = gs.GridSpec(2,1, hspace = hspace, height_ratios = [2,1])
g.update(left = 0.08, right = 0.26, wspace = 0.02)
a1 = plt.subplot(g[0])
im = plt.scatter(data_vaterite1[0], data_vaterite1[1]/1e6, s=7, c=data_vaterite1[2]*1e7)
plt.plot(fit_vaterite1[0], fit_vaterite1[1]/1e6, "r--", lw = 1.5)


plt.grid()
plt.xlim(-100, 1200)
plt.ylim(2.5, 6)
plt.ylabel(ylabel)
a1.yaxis.set_label_coords(-0.34, 0.5)

plt.xlabel("Time [s]")
plt.legend(loc="lower right", frameon = False)
plt.text(200, 3, r"$\mathcal{\tau} = 503\pm14$ s", size = 13, backgroundcolor='white')

plt.gca().set_xticklabels([])

a2 = plt.subplot(g[1])
plt.scatter(data_vaterite1[0], data_vaterite1[3]*100, s=7, c=data_vaterite1[2]*1e7)

plt.ylim(-0.3, 0.3)
plt.xlim(-100, 1200)
a2.yaxis.set_label_coords(-0.34, 0.5)
plt.ylabel("Residuals [%]")
plt.xlabel("Time [s]")
plt.grid()
plt.legend(loc="upper right", frameon = False)
plt.legend()
#plt.gcf().set_size_inches(6.4*2,5)
# fig1.subplots_adjust(right=0.8)
cbar_ax = fig1.add_axes([0.27, 0.15, 0.01, 0.82])
fig1.colorbar(im, cax=cbar_ax)
plt.subplots_adjust(right = 0.80, top = 0.98, left = 0.16, bottom = 0.13)






g = gs.GridSpec(2,1, hspace = hspace, height_ratios = [2,1])
g.update(left = 0.405, right = 0.405+0.18, wspace = 0.02)
plt.subplot(g[0])
im = plt.scatter(data_SiO2_2[0], data_SiO2_2[1]/1e6, s=7, c=data_SiO2_2[2]*1e7,  vmin=1.67, vmax=1.96)
plt.plot(fit_SiO2_2[0], fit_SiO2_2[1]/1e6, "r--", lw = 1.5)
plt.plot(fit_SiO2_2[0], fit_SiO2_2[2]/1e6, "r--", lw = 1.5)

plt.grid()
plt.ylim(0.3,0.850)
plt.xlim(-3000, 90000)
#plt.ylabel(ylabel)
plt.xlabel("Time [s]")
plt.legend(loc="lower right", frameon = False)
plt.xticks([0.0, 40000., 80000.])

plt.text(3900, 0.78, r"$\mathcal{\tau} = (3.0\pm0.2)\times10^{4} $ s", size = 13,  backgroundcolor='white')

plt.gca().set_xticklabels([])

plt.subplot(g[1])
plt.scatter(data_SiO2_2[0], data_SiO2_2[3]*100, s=7, c=data_SiO2_2[2]*1e7,  vmin=1.67, vmax=1.96)

plt.ylim(-0.6, 0.6)
plt.xlim(-3000, 90000)
#plt.ylabel("Residuals [%]")
plt.xlabel("Time [s]")
plt.grid()
plt.legend(loc="upper right", frameon = False)
plt.xticks([0.0, 40000., 80000.])
plt.legend()
plt.gcf().set_size_inches(13.5,5)
fig1.subplots_adjust(right=0.8)
cbar_ax = fig1.add_axes([0.595, 0.15, 0.01, 0.82])
fig1.colorbar(im, cax=cbar_ax)
plt.subplots_adjust(right = 0.750, top = 0.98, left = 0.12, bottom = 0.13)






g = gs.GridSpec(2,1, hspace = hspace, height_ratios = [2,1])
g.update(left = 0.73, right = 0.73+0.18, wspace = 0.02)
plt.subplot(g[0])
im = plt.scatter(data_SiO2_1[0], data_SiO2_1[1]/1e6, s=7, c=data_SiO2_1[2]*1e7)
plt.plot(fit_SiO2_1[0], fit_SiO2_1[1]/1e6, "r--", lw = 1.5)

plt.grid()
plt.ylim(0.5,5)
plt.xlim(-3000,105000)
#plt.ylabel(ylabel)
plt.xlabel("Time [s]")
plt.legend(loc="lower right", frameon = False)
plt.text(50, 4.45, r"$\mathcal{\tau} = (5.9\pm0.3)\times10^{4} $ s", size = 13, backgroundcolor='white')
plt.gca().set_xticklabels([])
plt.xticks([0.0, 40000, 80000])


plt.subplot(g[1])
plt.scatter(data_SiO2_1[0], data_SiO2_1[3]*100, s=7, c=data_SiO2_1[2]*1e7)

plt.ylim(-1.2,1.2)
plt.xlim(-3000,105000)
#plt.ylabel("Residuals [%]")
plt.xlabel("Time [s]")

plt.grid()
plt.legend(loc="upper right", frameon = False)
plt.xticks([0.0, 40000, 80000])
plt.legend()
plt.gcf().set_size_inches(6.4*2, 3.75)



cbar_ax = fig1.add_axes([0.92, 0.15, 0.01, 0.82])
fig1.colorbar(im, cax=cbar_ax, label = r"Pressure[$10^{-7}$mbar]", ticks=[0.95, 0.98, 1.01, 1.04, 1.07, 1.10])
cbar_ax.set_yticklabels(["0.95", "0.98", "1.01", "1.04", "1.07", "1.10"]) 

plt.subplots_adjust(right = 0.80, top = 0.97, left = 0.185, bottom = 0.15)

plt.savefig(r"C:\data\Rotation_paper\3plots.pdf")


plt.show()
