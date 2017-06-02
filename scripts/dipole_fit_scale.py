import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab

path = r'C:\data\20170511\bead2_15um_QWP\new_sensor_feedback\charge45_whole_points\60.0_74.9_150.0'
# path = r'C:\data\20170511\bead2_15um_QWP\new_sensor_feedback\charge44_whole_points'
file_name = 'ACamplitudes.txt'

distance = 0.002 #m

Vpp_to_Vamp = 0.5

trek = 200.0 # gain of the trek

epsilon0 = 8.85418782e-12 #m-3 kg-1 s4 A2

F = np.loadtxt(os.path.join(path, file_name))

Ea = trek*Vpp_to_Vamp*F[0]/distance

# Ed =  trek*Vpp_to_Vamp*0.1/distance
#g = dE/E

def Fw(X, p0, back):
    """fit dipole ac field only at freq w"""
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

def alpha_0(r): # in um
    """alpha0 , r is the radius in um"""
    r1 = 1.0*r/(1e6)
    epsilon0 = 8.854e-12
    return 3.*epsilon0*(2./5.)*(4.*np.pi/3.)*(r1**3)

def getmin_index(A):
    return np.argmin(A)
def order(A, B, C):
    return zip(*sorted(zip(A, B, C)))

def get_stuff_for_2W(path, file_name):
    F = np.loadtxt(os.path.join(path, file_name))
    Ea = trek * Vpp_to_Vamp * F[0] / distance
    Ea_order, force_W_order, force_2W_order = order(Ea, F[1], F[2])
    alpha0 = np.ones(len(Ea)) * alpha_0(7.5)
    popt_2W, pcov_2W = curve_fit(F2w, (Ea_order, alpha0), force_2W_order)
    error = np.sqrt(pcov_2W[0,0])
    return Ea_order, force_2W_order, popt_2W, alpha0, error

def plot_2W_curves(path, file_name):
    """ plot all curves with 'file_name' in each sub-folder of 'path' """
    #plot_2W_curves(path, file_name)
    file_list = glob.glob(path+"/*/"+file_name)
    N = len(file_list)
    colormap = get_color_map(N)
    plt.figure()
    for name, color in zip(file_list, colormap):
        file_label = name[len(path):name.rfind(file_name)]
        pathname = name[:name.rfind(file_name)]
        Ea_order, force_2W_order, popt_2W, alpha0, error = get_stuff_for_2W(pathname, file_name)
        plt.plot(Ea_order, force_2W_order, ".", color = color, label = file_label)
        plt.plot(Ea_order, F2w((np.array(Ea_order), np.array(alpha0)), *popt_2W), color = color)
    plt.ylabel("Force (N)")
    plt.xlabel("AC field amplitude (N/e)")
    plt.legend()
    plt.title(path[path.rfind('\\'):])
    plt.show()

def plot_2W_Force_vs_X(path, file_name):
    """ plot all curves with 'file_name' in each sub-folder of 'path' """
    #plot_2W_Force_vs_X(path, file_name)
    file_list = glob.glob(path+"/*/"+file_name)
    N = len(file_list)
    X = np.zeros(N)
    qForce = np.zeros(N)
    error = np.zeros(N)
    for index in range(N):
        name = file_list[index]
        file_label = name[len(path):name.rfind(file_name)]
        X[index] = float(file_label[1:file_label.find('.')])
        pathname = name[:name.rfind(file_name)]
        Ea_order, force_2W_order, popt_2W, alpha0, err = get_stuff_for_2W(pathname, file_name)
        error[index] = float(err)
        qForce[index] = popt_2W[0]
    X, qForce = zip(*sorted(zip(X, qForce)))
    fit = np.polyfit(X, qForce, 2)
    plt.figure()
    plt.errorbar(X, qForce, yerr = error, fmt = 'o', label = 'data')
    plt.plot(X, np.polyval(fit, X), label = 'fit')
    plt.ylabel("Force (N)")
    plt.xlabel("Piezo X position [Volts]")
    plt.legend()
    plt.title(path[path.rfind('\\'):])
    plt.show()

def plot_2W_Force_vs_Z(path, file_name):
    """ plot all curves with 'file_name' in each sub-folder of 'path' """
    file_list = glob.glob(path+"/*/"+file_name)
    N = len(file_list)
    Z = np.zeros(N)
    qForce = np.zeros(N)
    error = np.zeros(N)
    for index in range(N):
        name = file_list[index]
        file_label = name[len(path):name.rfind(file_name)-1]
        Z[index] = float(file_label[file_label.rfind('_')+1:])
        pathname = name[:name.rfind(file_name)]
        Ea_order, force_2W_order, popt_2W, alpha0, err = get_stuff_for_2W(pathname, file_name)
        error[index] = float(err)
        qForce[index] = popt_2W[0]
    Z, qForce = zip(*sorted(zip(Z, qForce)))
    fit = np.polyfit(Z, qForce, 2)
    plt.figure()
    plt.errorbar(Z, qForce, yerr = error, fmt = 'o', label = 'data')
    plt.plot(Z, np.polyval(fit, Z), label = 'fit')
    plt.ylabel("Force (N)")
    plt.xlabel("Piezo Z position [Volts]")
    plt.legend()
    plt.title(path[path.rfind('\\'):])
    plt.show()

alpha0 = np.ones(len(Ea))*alpha_0(7.5)

order(Ea, F[1], F[2])

popt_2W, pcov_2W = curve_fit(F2w, (Ea_order, alpha0), force_2W_order)

g_from_fit = np.ones(len(Ea))*popt_2W[0]

popt_W, pcov_W = curve_fit(Fw, (Ea_order, g_from_fit), force_W_order)



plt.figure()
plt.loglog(Ea_order, force_W_order, ".")
plt.loglog(Ea_order, force_2W_order, ".")
plt.loglog(Ea_order, Fw((np.array(Ea_order),np.array(g_from_fit)), *popt_W))
plt.loglog(Ea_order, F2w((np.array(Ea_order),np.array(alpha0)), *popt_2W))

plt.ylabel("Force (N)")
plt.xlabel("AC field amplitude (N/e)")
plt.title(path[path.rfind('\\'):])
plt.show()
