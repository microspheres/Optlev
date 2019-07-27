import os, glob
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

"""Files"""
path = r'/data/20170511/bead2_15um_QWP/new_sensor_feedback/charge43_whole_points'
file_name = 'ACamplitudes.txt'

"""Constants"""
distance = 0.002  # m
Vpp_to_Vamp = 0.5
trek = 200.0  # gain of the trek
epsilon0 = 8.85418782e-12 #m-3 kg-1 s4 A2
Ed =  trek*Vpp_to_Vamp*0.1/distance
# g = dE/E


def Fw(X, p0, back):
    """fit dipole ac field only at freq w"""
    Eac, g = X
    return p0 * Eac * g + back

def F2w(X, g, back):
    """fit dipole ac field only at freq 2w"""
    Eac, alpha = X
    return alpha * (Eac ** 2) * g + back

def Fwacdc(X, g, p0, back, alpha):
    """fit dipole ac and dc field only at freq w"""
    Eac, Edc = X
    return p0 * Eac * g + back + alpha * (2.0 * g * Edc * Eac)

def alpha_0(r):  # r in um
    """returns alpha0
       r is the radius in um"""
    r1 = r / (1e6) # meters
    epsilonr = 3.
    return 3. * epsilon0 * ((epsilonr - 1)/(epsilonr + 2)) * (4. * np.pi / 3.) * (r1 ** 3)


def order(A, B, C):
    return zip(*sorted(zip(A, B, C)))

def get_color_map(n):
    """returns a color map of length n"""
    jet = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=0, vmax=n-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    outmap = []
    for i in range(n):
        outmap.append( scalarMap.to_rgba(i) )
    return outmap

def get_F_Ea_stuff(path, file_name):
    """returns [Ea, Ea_order, force_W_order,
                force_2W_order, popt_2W, pcov_2W,
                g_from_fit, popt_W, pcov_W]"""
    F = np.loadtxt(os.path.join(path, file_name))
    Ea = trek * Vpp_to_Vamp * F[0] / distance
    Ea_order, force_W_order, force_2W_order = order(Ea, F[1], F[2])
    alpha0 = np.ones(len(Ea)) * alpha_0(7.5)
    popt_2W, pcov_2W = curve_fit(F2w, (Ea_order, alpha0), force_2W_order)
    g_from_fit = np.ones(len(Ea)) * popt_2W[0]
    popt_W, pcov_W = curve_fit(Fw, (Ea_order, g_from_fit), force_W_order)
    return [Ea, Ea_order, force_W_order, force_2W_order, popt_2W, pcov_2W, g_from_fit, popt_W, pcov_W]


def get_alpha0_g_p0_bg(path, file_name):
    a = get_F_Ea_stuff(path, file_name)
    popt_2W = a[4]
    popt_W = a[7]
    return alpha_0(7.5), popt_2W[0], popt_W[0], popt_W[1]

def plot_force_W_2W(path, file_name):
    a = get_F_Ea_stuff(path, file_name)
    Ea = a[0]
    Ea_order = a[1]
    force_W_order = a[2]
    force_2W_order = a[3]
    g_from_fit = a[6]
    popt_W = a[7]
    popt_2W = a[4]
    alpha0 = np.ones(len(Ea)) * alpha_0(7.5)
    plt.figure()
    plt.plot(Ea_order, force_W_order, ".")
    plt.plot(Ea_order, force_2W_order, ".")
    plt.plot(Ea_order, Fw((np.array(Ea_order), np.array(g_from_fit)), *popt_W))
    plt.plot(Ea_order, F2w((np.array(Ea_order), np.array(alpha0)), *popt_2W))
    plt.ylabel("Force (N)")
    plt.xlabel("AC field amplitude (N/e)")
    plt.title(path[path.rfind('\\'):])
    plt.show()

def loglog_force_W_2W(path, file_name):
    a = get_F_Ea_stuff(path, file_name)
    Ea = a[0]
    Ea_order = a[1]
    force_W_order = a[2]
    force_2W_order = a[3]
    g_from_fit = a[6]
    popt_W = a[7]
    popt_2W = a[4]
    alpha0 = np.ones(len(Ea)) * alpha_0(7.5)
    plt.figure()
    plt.loglog(Ea_order, force_W_order, ".")
    plt.loglog(Ea_order, force_2W_order, ".")
    plt.loglog(Ea_order, Fw((np.array(Ea_order), np.array(g_from_fit)), *popt_W))
    plt.loglog(Ea_order, F2w((np.array(Ea_order), np.array(alpha0)), *popt_2W))
    plt.ylabel("Force (N)")
    plt.xlabel("AC field amplitude (N/e)")
    plt.title(path[path.rfind('\\'):])
    plt.show()

def get_stuff_for_2W(file_name):
    F = np.loadtxt(file_name)
    Ea = trek * Vpp_to_Vamp * F[0] / distance
    Ea_order, force_W_order, force_2W_order = order(Ea, F[1], F[2])
    alpha0 = np.ones(len(Ea)) * alpha_0(7.5)
    popt_2W, pcov_2W = curve_fit(F2w, (Ea_order, alpha0), force_2W_order)
    return Ea_order, force_2W_order, popt_2W, alpha0


def plot_2W_curves(path, file_name):
    """ plot all curves with 'file_name' in each sub-folder of 'path' """
    file_list = glob.glob(path+"/*/"+file_name)
    N = len(file_list)
    colormap = get_color_map(N)
    plt.figure()
    for name, color in zip(file_list, colormap):
        file_label = name[len(path):name.rfind(file_name)]
        print file_label
        Ea_order, force_2W_order, popt_2W, alpha0 = get_stuff_for_2W(name)
        plt.plot(Ea_order, force_2W_order, ".", color = color, label = file_label)
        plt.plot(Ea_order, F2w((np.array(Ea_order), np.array(alpha0)), *popt_2W), color = color)
    plt.ylabel("Force (N)")
    plt.xlabel("AC field amplitude (N/e)")
    plt.title(path[path.rfind('\\'):])
    plt.legend()
    plt.show()

plot_2W_curves(path, file_name)
