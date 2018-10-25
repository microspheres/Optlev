import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

matplotlib.rcParams.update({'font.size': 16})



### Data for SiO2 10um spheres
P_data = [2.12e-6, 1.76e-7, 5.41e-7]
xerr = [ np.sqrt(0.008**2 + 0.3**2)*2.12e-6, np.sqrt(0.035**2 + 0.3**2)*1.76e-7, np.sqrt(0.019**2 + 0.3**2)*5.41e-7]
Tau_data = [2192., 30100., 9685.]
yerr = [np.sqrt((0.008)**2 + (18./2192.)**2)*2192., np.sqrt((0.035)**2 + (1950/30100)**2)*30100, np.sqrt((0.019)**2 + (280./9685.)**2)*9685] ## inside the sqrt is the error due to pressure and from the least square method repectively. The sigma used for the least square method comes from max of pressure fluctation (dp/p * tau) or residuals.

# dia = 10.3
r = 5.15e-6 # m
rho = 1800.0 # kg/m3
kb = 1.380648e-23 # SI
T = 293.0 # K

m =  (2./28.)*4.65e-26  # H2 molecule mass in kg

dr = 0.136*r
drho = 0.*rho




### Data for SiO2 10um spheres ____ 2
P_data2 = [1.80e-6, 1.05e-7, 5.71e-7, 5.04e-6]
xerr2 = [np.sqrt(0.3**2 + 0.012**2)*1.8e-6, np.sqrt(0.3**2 + 0.05**2)*1.05e-7, np.sqrt(0.3**2 + 0.026**2)*5.71e-7, np.sqrt(0.3**2 + 0.033**2)*5.04e-6]
Tau_data2 = [2249., 59400., 9740., 784.]
yerr2 = [ np.sqrt(0.012**2 + (30./2249.)**2)*2249., np.sqrt(0.05**2 + (1500./59400.)**2)*59400, np.sqrt(0.026**2 + (50./9740.)**2)*9740., np.sqrt(0.033**2 + (20./784.)**2)*784.]


# dia = 10.3
r2 = 5.15e-6 # m
rho2 = 1800.0 # kg/m3
kb = 1.380648e-23 # SI
T2 = 293.0 # K


dr2 = 0.136*r
drho2 = 0.*rho




##### Data for Vaterite
P_datav = [2.24e-4, 1.43e-5]
xerrv =[np.sqrt(0.3**2 + 0.011**2)*2.24e-4, np.sqrt(0.3**2 + 0.019**2)*1.43e-5]
Tau_datav = [25.5, 503.0]
yerrv = [np.sqrt(0.011**2 + (0.5/25)**2)*25, np.sqrt(0.011**2 + (13/503)**2)*503]

diav = 5.3
rv = 2.65e-6 # m
rhov = 2450.0 # kg/m3
kbv = 1.380648e-23 # SI
Tv = 293.0 # k
mv = 4.65e-26 # N2 molecule mass in kg, only true for the vaterite

drv = 0.1*r
drhov = 0.

#######################################################
######################################################
####################################################



pi = np.pi

def mean_speed(T,m):
    a = np.sqrt(8.0*kb*T/(pi*m)) # mean (arithmetic) speed
    # a = np.sqrt(3.0*kb*T/(m)) # rms speed
    return a

def spinuptime(p,r,rho,T,m): # p is in mbar
    p = p*100.
    a = 0.1*pi*r*rho*(mean_speed(T,m))/p
    return a

def spinuptime_err(p,r,rho,T,m, dr, drho): # p is in mbar
    p = p*100.
    err = (0.1*pi*r*rho*(mean_speed(T,m))/p)*np.sqrt( (dr/r)**2 + (drho/rho)**2 )
    return err

def linear(x, a):
    return a/x

def linear2(x, a):
    return -a*x

print (mean_speed(T,m))

### For SiO2
P = np.arange(1e-8, 1e-5, 1e-8)
Tau = spinuptime(P,r,rho,T,m)
Tau_err = spinuptime_err(P,r,rho,T,m, dr, drho)
popt, pcov = curve_fit(linear, P_data, Tau_data, sigma = 0.3*np.array(Tau_data))

### For SiO2_______2
P2 = np.arange(1e-8, 1e-5, 1e-8)
popt2, pcov2 = curve_fit(linear, P_data2, Tau_data2, sigma = 0.3*np.array(Tau_data2))

### For Vaterite
Pv = np.arange(1e-5, 1e-3, 1e-5)
Tauv = spinuptime(Pv,rv,rhov,Tv,mv)
Tau_errv = spinuptime_err(Pv,rv,rhov,Tv,mv, drv, drhov)
poptv, pcovv = curve_fit(linear, P_datav, Tau_datav, sigma = 0.3*np.array(Tau_datav))


print "vaterite damping at 4e-2 mbar:", spinuptime(4e-2,rv,rhov,Tv,mv)



rho = 2450.0 # SI
radius = (5.0/2.)*(1e-6) # m
Iner = (8./15.)*rho*np.pi*(radius**5)
print "vaterite torque 4e-2mbar", Iner*(2*np.pi*34900)/(0.24)







plt.figure()
### SiO2 10um
# plt.plot(P, Tau, label = "Model SiO2 spheres")
plt.fill_between(P, Tau + Tau_err, Tau - Tau_err, alpha = 0.5)
plt.errorbar(P_data, Tau_data, xerr=xerr, yerr = yerr, fmt='ro', label = "10$\mu m$ SiO$_2$ sphere #1")
plt.loglog(P, linear(P, *popt), "r--", lw = 1)

### SiO2 10um____2
plt.errorbar(P_data2, Tau_data2, xerr=xerr2, yerr = yerr2, fmt='ko', label = "10$\mu m$ SiO$_2$ sphere #2")
plt.plot(P2, linear(P2, *popt2), "k--", lw = 1)

### vaterite
# plt.loglog(Pv, Tauv, label = "Model Vaterite spheres")
plt.fill_between(Pv, Tauv + Tau_errv, Tauv - Tau_errv,alpha = 0.5)
plt.errorbar(P_datav, Tau_datav, xerr=xerrv, yerr = yerrv, fmt='bx', label = "5$\mu m$ Vaterite sphere")
plt.plot(Pv, linear(Pv, *poptv), "b--", lw = 1)


plt.ylim(.05, 1e6)
plt.xlim(6e-8, 1e-1)
plt.legend(loc = "lower left")
plt.ylabel("Damping time [s]", fontsize = 16)
plt.xlabel("Pressure [mbar]", fontsize = 16)
plt.rcParams.update({'font.size': 14})
plt.tight_layout(pad = 0)
plt.grid()

plt.savefig(r"C:\data\Rotation_paper\Figures\damping_time_vs_pressure.pdf")


############################ subplot


import matplotlib.gridspec as gs
matplotlib.rcParams.update({'font.size': 16})

fig = plt.figure()

g = gs.GridSpec(1, 2, wspace = 0.3)
plt.subplot(g[0])

### SiO2 10um
# plt.plot(P, Tau, label = "Model SiO2 spheres")
plt.fill_between(P, Tau + Tau_err, Tau - Tau_err, alpha = 0.5)
plt.errorbar(P_data, Tau_data, xerr=xerr, yerr = yerr, fmt='ro', label = "SiO$_2$ #1")
plt.loglog(P, linear(P, *popt), "r--", lw = 1)

### SiO2 10um____2
plt.errorbar(P_data2, Tau_data2, xerr=xerr2, yerr = yerr2, fmt='ko', label = "SiO$_2$ #2")
plt.plot(P2, linear(P2, *popt2), "k--", lw = 1)

plt.grid()
plt.gca().set_xticks([1e-7, 1e-6, 1e-5])
plt.gca().set_xticklabels(["$10^{-7}$","$10^{-6}$", "$10^{-5}$"])
plt.xlim(6e-8, 1e-5)
plt.ylim(10, 1e6)

plt.ylabel("Damping time [s]")

legend = plt.legend(loc="lower left", frameon = True)
legend.get_frame().set_facecolor('#FFFFFF')
legend.get_frame().set_alpha(1)



plt.subplot(g[1])

plt.fill_between(Pv, Tauv + Tau_errv, Tauv - Tau_errv, alpha = 0.5, color = "r")
plt.errorbar(P_datav, Tau_datav, xerr=xerrv, yerr = yerrv, fmt='bs', label = "Vaterite")
plt.loglog(Pv, linear(Pv, *poptv), "b--", lw = 1)

plt.gca().set_xticks([1e-5, 1e-4, 1e-3])
plt.gca().set_xticklabels(["$10^{-5}$","$10^{-4}$","$10^{-3}$"])
plt.gca().set_yticklabels([])
plt.ylim(10, 1e6)
plt.xlim(1e-5, 1e-3)

plt.grid()
legend = plt.legend(loc="upper right", frameon = True)
legend.get_frame().set_facecolor('#FFFFFF')
legend.get_frame().set_alpha(1)

fig.subplots_adjust(right=0.8)
plt.subplots_adjust(right = 0.95, top = 0.96, left = 0.13, bottom = 0.18)
fig.text(0.5, 0.01, "Pressure [mbar]", ha='center')
plt.gcf().set_size_inches(6.4, 3.3)

plt.savefig("C:\data\Rotation_paper\Figures\damping_time_vs_pressure_sub.pdf")



plt.show()
