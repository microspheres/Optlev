import numpy as np
import matplotlib.pyplot as plt


P_data = [2.24e-4, 1.45e-5]
xerr = [1.0e-5, 1.0e-6]
Tau_data = [26.0, 506.0]
yerr = [0.4, 0.4]

dia = 5.3
r = 2.65e-6 # m
rho = 2450.0 # kg/m3
kb = 1.380648e-23 # SI
T = 293.0 # k
m = 4.65e-26 # N2 molecule mass in kg

dr = 0.1*r
drho = 0.

pi = np.pi

def mean_speed(T,m):
    a = np.sqrt(8.0*kb*T/(pi*m))
    return a
    

def spinuptime(p,r,rho,T,m): # p is in mbar
    p = p*100.
    a = 0.1*pi*r*rho*(mean_speed(T,m))/p
    return a

def spinuptime_err(p,r,rho,T,m, dr, drho): # p is in mbar
    p = p*100.
    err = (0.1*pi*r*rho*(mean_speed(T,m))/p)*np.sqrt( (dr/r)**2 + (drho/rho)**2 )
    return err

print (mean_speed(T,m))

P = np.arange(1e-7, 1e-3, 1e-6)

Tau = spinuptime(P,r,rho,T,m)

Tau_err = spinuptime_err(P,r,rho,T,m, dr, drho)

Tau2 = spinuptime(P,r,0.8*rho,T,m)

Tau_err2 = spinuptime_err(P,r,0.8*rho,T,m, dr, drho)

plt.figure()
plt.loglog(P, Tau, label = "100% of the density")
plt.loglog(P, Tau2, label = "80% of the density")
plt.fill_between(P, Tau + Tau_err, Tau - Tau_err, alpha = 0.5)
plt.fill_between(P, Tau2 + Tau_err2, Tau2 - Tau_err2, alpha = 0.5)

plt.errorbar(P_data, Tau_data, xerr=xerr, yerr = yerr, fmt='ro')

plt.legend(loc=0)
plt.ylabel("Damping time [s]", fontsize = 16)
plt.xlabel("Pressure [mbar]", fontsize = 16)
plt.rcParams.update({'font.size': 14})
plt.tight_layout(pad = 0)
plt.grid()


############################ subplot


import matplotlib.gridspec as gs

fig = plt.figure()

g = gs.GridSpec(1,2, vspace = 0.07)

plt.subplot(g[0])

im = plt.scatter(c1[2][good_points1] - 320, c1[0][good_points1]/1e6, s=8, c=c1[1][good_points1]*1e5)

plt.plot(times - 320, func(times, *popt)/1e6, "r--", lw = 1.5)


plt.grid()
plt.xlim(-100, 1200)
plt.ylim(2.5, 6)

plt.ylabel("Rotation [MHz]")
plt.xlabel("Time [s]")


plt.legend(loc="lower right", frameon = False)
plt.gca().set_xticklabels([])




plt.subplot(g[1])

plt.scatter(c1[2][good_points1] - 320, 100*(func(c1[2][good_points1], *popt) -  c1[0][good_points1])/c1[0][good_points1], s=8, c=(c1[1][good_points1])*1e5)


plt.ylim(-0.3, 0.3)
plt.xlim(-100, 1200)
plt.ylabel("Residuals [%]")
plt.xlabel("Time [s]")
# plt.colorbar()
plt.grid()
plt.legend(loc="upper right", frameon = False)


plt.legend()

plt.gcf().set_size_inches(6.4,5)
# plt.tight_layout()

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.81, 0.1, 0.03, 0.88])
fig.colorbar(im, cax=cbar_ax, label = r"Pressure [$\times 10^{-5}$mbar]")

plt.subplots_adjust(right = 0.80, top = 0.98, left = 0.16, bottom = 0.13)

plt.savefig("paper_plot.pdf")



plt.show()
