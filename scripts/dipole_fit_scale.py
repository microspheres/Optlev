import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab
from scipy.stats import linregress

#folder = '/data/20170511/bead2_15um_QWP/new_sensor_feedback/charge43_whole_points'
#path = folder + '/60.0_74.9_75.4'

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

def order2(A,B):
    return zip(*sorted(zip(A,B)))

def pull_param(path, useDC = False):
    """pulls out the AC voltage, force at f, and force at 2f from the given path if a file is already in it"""
    
    if useDC:
        file_name = 'ACandDCamplitudes.txt'
    else:
        file_name = 'ACamplitudes.txt'

    F = np.loadtxt(os.path.join(path, file_name)) # gives voltages in V and amplitudes in N
    
    return F[0], F[1], F[2]
    

def get_param(ACvoltages, omegaAmplitudes, twoOmegaAmplitudes):
    """takes in AC voltage vs force at the drive frequency (f) and twice the drive frequency (2f)
       outputs the electric field, the measured forces at f and 2f, and the fitted forces at f and 2f"""
    ACvoltage = np.array(ACvoltages)
    force_f = np.array(omegaAmplitudes)
    force_2f = np.array(twoOmegaAmplitudes)
    Ea = trek*Vpp_to_Vamp*ACvoltage/distance # V/m

    alpha0 = np.ones(len(Ea))*alpha_0(7.5) # Nm^3/V^2
    Ea_order, force_W_order, force_2W_order = order(Ea, force_f, force_2f) # V/m, N, N
    popt_2W, pcov_2W = curve_fit(F2w, (Ea_order, alpha0), force_2W_order)
    g_from_fit = np.ones(len(Ea))*popt_2W[0] # m^-1
    popt_W, pcov_W = curve_fit(Fw, (Ea_order, g_from_fit), force_W_order)
    err_p0 = np.sqrt(np.diag(pcov_W))[0]
    err_back_W = np.sqrt(np.diag(pcov_W))[1]
    err_g = np.sqrt(np.diag(pcov_2W))[0]
    err_back_2W = np.sqrt(np.diag(pcov_2W))[1]

    alpha0 = alpha_0(7.5) # Nm^3/V^2
    g = popt_2W[0] # m^-1
    error_g = np.sqrt(pcov_2W[0][0]) # m^-1
    p0 = popt_W[0] # Nm^2/V
    background = popt_W[1] # N

    print 'Linear and quadratic parameters'
    print '    alpha = '+str(alpha0)+' Nm^3/V^2'
    print '    g = '+str(g)+' m^-1'
    print '    error of g = '+str(error_g)+' m^-1'
    print '    p0 = '+str(p0)+' Nm^2/V'
    print '    background = '+str(background)+' N'

    err_f = 5e-17
    err_2f = 5e-17
    #err_f = np.sqrt((p0*Ea*g*np.sqrt((err_p0/p0)**2 + (err_g/g)**2))**2 + err_back_W**2) # p0*Eac*g + back
    #err_2f = np.sqrt((alpha0*(Ea**2)*err_g)**2 + err_back_2W**2)

    Efield = Ea_order # V/m
    data_f = force_W_order # N
    data_2f = force_2W_order # N
    fit_f = Fw((np.array(Ea_order),np.array(g_from_fit)), *popt_W) # N
    fit_2f = F2w((np.array(Ea_order),np.array(alpha0)), *popt_2W) # N

    return Efield, data_f, data_2f, fit_f, fit_2f, err_f, err_2f

#def plot_amplitude_data_raw(path, Efield, data_f, data_2f, fit_f, fit_2f, err_f, err_2f):
#    plt.figure()
#    plt.plot(Efield, data_f, ".")
#    plt.plot(Efield, data_2f, ".")
#    plt.errorbar(Efield, fit_f, yerr = err_f)
#    plt.errorbar(Efield, fit_2f, yerr = err_2f)

#    plt.ylabel("Force (N)")
#    plt.xlabel("AC field amplitude (N/e)")
#    plt.title(path[path.rfind('\\'):])
#    plt.show(block = False)

def plot_amplitude_data_raw(path, Efield, data_f, data_2f, fit_f, fit_2f, err_f, err_2f):
    """Assuming Efield is given in V/m"""
    Efield = np.array(Efield)/1e6
    #print data_f
    #print data_2f
    plt.figure()
    plt.errorbar(Efield, data_f, yerr = err_f, label = 'permanent dipole', fmt = "b.")
    plt.plot(Efield, fit_f, 'b') # permanent dipole
    plt.errorbar(Efield, data_2f, yerr = err_2f, label = 'induced dipole', fmt = "r.")
    plt.plot(Efield, fit_2f, 'r') # induced dipole
    plt.xlim([-0.03, 0.03+Efield[-1]])

    plt.xlabel("AC field amplitude [kV/mm]")
    plt.ylabel("Force [N]")
    plt.legend(loc = 2)
    plt.title("Permanent and induced dipole force")
    plt.show(block = False)

def plot_amplitude_data(path, useDC = False):
    ACvoltages, omegaAmplitudes, twoOmegaAmplitudes = pull_param(path, useDC)
    Efield, data_f, data_2f, fit_f, fit_2f, err_f, err_2f = get_param(ACvoltages, omegaAmplitudes, twoOmegaAmplitudes)
    plot_amplitude_data_raw(path, Efield, data_f, data_2f, fit_f, fit_2f, err_f, err_2f)

def para(x, a, b, c):
    return a*(x-b)**2 + c

def plot_highest_point(f, data_err, useDC = False):
    """takes in AC voltage vs force at twice the drive frequency (f)
       from every sub-folder in the given folder
       plots the x/y/z position of the piezo vs the highest force"""

    folders = next(os.walk(f))[1]
    
    if useDC:
        file_name = 'ACandDCamplitudes.txt'
    else:
        file_name = 'ACamplitudes.txt'

    N = len(folders)
    x_pos =[]
    force = []

    for folder in folders:
        x_index = folder.find('_')
        x_pos.append(float(folder[0:x_index]))
            
        F = np.loadtxt(os.path.join(f, folder, file_name)) # gives voltages in V and amplitudes in N
        #Efield = trek*Vpp_to_Vamp*F[0]/distance # V/m
        #force_f = F[1]
        #force_2f = F[2]
        force.append(float(max(F[2])))
        print str(max(F[2]))

    x_pos, force = order2(x_pos, force)
    popt, pcov = curve_fit(para, x_pos, force)
    a = popt[0]
    b = popt[1]
    c = popt[2]
    force_fit = para(x_pos, a, b, c)
    err_b = np.sqrt(np.diag(pcov))[1]
    err_c = np.sqrt(np.diag(pcov))[2]
    
    plt.figure()
    #plt.errorbar(x_pos, force, yerr = 2.*data_err, fmt = 'o')
    plt.plot(x_pos, force, 'o')
    plt.plot(x_pos, force_fit)
    plt.xlim([-3, 3+x_pos[-1]])
    #plt.ylim([2.35e-15,3.25e-15])
    plt.xlabel("X position of piezo [V]")
    plt.ylabel("Maximum force measured [N]")
    plt.title("Force vs Piezo position")
    plt.show(block = False)

def line(x, a, b):
    return a*x + b

def plot_highest_point_z(f, data_err, useDC = False):
    """takes in AC voltage vs force at twice the drive frequency (f)
       from every sub-folder in the given folder
       plots the z position of the piezo vs the highest force"""

    folders = next(os.walk(f))[1]
    
    if useDC:
        file_name = 'ACandDCamplitudes.txt'
    else:
        file_name = 'ACamplitudes.txt'

    N = len(folders)
    z_pos =[]
    force = []

    for folder in folders:
        x_index = folder.find('_')
        y_index = folder.find('_', x_index+1)
        z_index = folder.find('_', y_index+1)
        z_pos.append(float(folder[y_index+1:z_index]))
            
        F = np.loadtxt(os.path.join(f, folder, file_name)) # gives voltages in V and amplitudes in N
        #Efield = trek*Vpp_to_Vamp*F[0]/distance # V/m
        #force_f = F[1]
        #force_2f = F[2]
        force.append(float(max(F[2])))

    z_pos, force = order2(z_pos, force)
    m, b, r, p, err = linregress(z_pos, force)

    force_fit = m*np.array(z_pos) + b
    
    plt.figure()
    #plt.errorbar(z_pos, force, yerr = data_err, fmt = 'o')
    plt.plot(z_pos, force, 'o')
    plt.plot(z_pos, force_fit)
    plt.xlim([-3, 3+z_pos[-1]])
    plt.xlabel("Z position of piezo [V]")
    plt.ylabel("Maximum force measured [N]")
    plt.title("Force vs Piezo position")
    plt.show(block = False)
