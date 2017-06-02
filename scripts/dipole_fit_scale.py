import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab

#path = '/data/20170511/bead2_15um_QWP/new_sensor_feedback/charge43_whole_points/60.0_74.9_75.4'

distance = 0.002 #m
Vpp_to_Vamp = 0.5 # files are saved in Vpp
trek = 200.0 # gain of the trek

def Fw(X, p0, back):
    """fit dipole ac field only at freq w
       Fw in N
       p0 in N m^2 / V
       g in m^-1"""
    Eac, g = X
    return p0*Eac*g + back

def F2w(X, g, back):
    """fit dipole ac field only at freq 2w"""
    Eac, alpha = X
    return alpha*(Eac**2)*g + back

def Fwacdc(X, g, p0, back, alpha):
    """fit dipole ac and dc field only at freq w"""
    Eac, Edc = X
    return p0*Eac*g + back + alpha*(2.0*g*Edc*Eac)

def alpha_0(r):
    """alpha0 N m^3 / V^2
       r is the radius in um"""
    r1 = 1.0*r/(1e6)
    epsilon0 = 8.854e-12
    return 3.*epsilon0*(2./5.)*(4.*np.pi/3.)*(r1**3)

def order(A,B,C):
    return zip(*sorted(zip(A,B,C)))

def get_param(path, useDC = False):
    """takes in AC voltage vs force at the drive frequency (f) and twice the drive frequency (2f)
       outputs the electric field, the measured forces at f and 2f, and the fitted forces at f and 2f"""
    
    if useDC:
        file_name = 'ACandDCamplitudes.txt'
    else:
        file_name = 'ACamplitudes.txt'

    F = np.loadtxt(os.path.join(path, file_name)) # gives voltages in V and amplitudes in N
    Ea = trek*Vpp_to_Vamp*F[0]/distance # V/m

    alpha0 = np.ones(len(Ea))*alpha_0(7.5) # Nm^3/V^2
    Ea_order, force_W_order, force_2W_order = order(Ea, F[1], F[2]) # V/m, N, N
    popt_2W, pcov_2W = curve_fit(F2w, (Ea_order, alpha0), force_2W_order)
    g_from_fit = np.ones(len(Ea))*popt_2W[0] # m^-1
    popt_W, pcov_W = curve_fit(Fw, (Ea_order, g_from_fit), force_W_order)
    err_p0 = np.sqrt(np.diag(pcov_W))[0]
    err_back_W = np.sqrt(np.diag(pcov_W))[1]
    err_g = np.sqrt(np.diag(pcov_2W))[0]
    err_back_2W = np.sqrt(np.diag(pcov_2W))[1]

    # alpha0 = alpha_0(7.5) # Nm^3/V^2
    g = popt_2W[0] # m^-1
    # error g = np.sqrt(pcov_2W[0][0]) # m^-1
    p0 = popt_W[0] # Nm^2/V
    # background = popt_W[1] # N

    err_f = np.sqrt((p0*Ea*g*np.sqrt((err_p0/p0)**2 + (err_g/g)**2))**2 + err_back_W**2) # p0*Eac*g + back
    err_2f = np.sqrt((alpha0*(Ea**2)*err_g)**2 + err_back_2W**2)

    Efield = Ea_order # V/m
    data_f = force_W_order # N
    data_2f = force_2W_order # N
    fit_f = Fw((np.array(Ea_order),np.array(g_from_fit)), *popt_W) # N
    fit_2f = F2w((np.array(Ea_order),np.array(alpha0)), *popt_2W) # N

    return Efield, data_f, data_2f, fit_f, fit_2f, err_f, err_2f

def plot_amplitude_data_raw(path, Efield, data_f, data_2f, fit_f, fit_2f, err_f, err_2f):
    plt.figure()
    plt.plot(Efield, data_f, ".")
    plt.plot(Efield, data_2f, ".")
    plt.errorbar(Efield, fit_f, yerr = err_f)
    plt.errorbar(Efield, fit_2f, yerr = err_2f)

    plt.ylabel("Force (N)")
    plt.xlabel("AC field amplitude (N/e)")
    plt.title(path[path.rfind('\\'):])
    plt.show(block = False)

def plot_amplitude_data(path, useDC = False):
    Efield, data_f, data_2f, fit_f, fit_2f, err_f, err_2f = get_param(path, useDC)
    plot_amplitude_data_raw(path, Efield, data_f, data_2f, fit_f, fit_2f, err_f, err_2f)




"""ratio between F2w and Fw is alpha*E/p0 and alpha/p0 is constant so can calculate that line
   want precision to make background larger than F2w
   then increase amount of time it takes to do the measurement so that the background decreases
       (so multiply the time by (alpha*E/p0)^2 for each value of E
   then do this same measurement all over again

   So this code should plot all the PSDs of one run on top of each other
   then look at the drive frequency for this run and integrate around the peak to get the corresponding force at f
   and then integrate around twice that frequency to get the corresponding force at 2f
   and finally plot the two against the AC voltage of the respective PSD to see the linear and quadratic behavior
   and get the values of p0, alpha, g, and the background."""
