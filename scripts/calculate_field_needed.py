import glob
import numpy as np
import matplotlib.pyplot as plt
from VoltagevsAmplitude import getACAmplitudeGraphs
from dipole_fit_scale import get_param, plot_amplitude_data_raw, plot_highest_point, plot_highest_point_z
from scipy.stats import linregress
from scipy.optimize import curve_fit

directory = '/data/20170511/bead2_15um_QWP/new_sensor_feedback'
folder = directory + '/charge43_whole_points'
folder2 = directory + '/charge45_whole_points'
path = folder + '/60.0_74.9_75.4'
# measured by eye to get background
background = 4e-17 # N

# First graph: plot PSDs on top of each other
file_list = glob.glob(path+"/*.h5")
ACvoltages, omegaAmplitudes, twoOmegaAmplitudes, DCvoltages = getACAmplitudeGraphs(file_list, make_plots = True, zeroDC = True)

"""The following two parameters will be calculated later."""
#max_limit = f(E) # N; we want the amplitude of the force at the drive frequency to be below this number
#desired_ratio = background/max_limit # we want to be able to see the peak at 2f so we know what the amplitude of the peak at f is

Efield, data_f, data_2f, fit_f, fit_2f, err_f, err_2f = get_param(ACvoltages, omegaAmplitudes, twoOmegaAmplitudes)

data_ratio = np.divide(data_2f, data_f) # unitless
#fit_ratio = np.divide(fit_2f, fit_f) # unitless

# here I'm estimating the error bars
binF = 0.0762939453125 # got this from VoltagevsAmplitude.py
conversion = 3.7139625927e-13
data_error = 4e-4*conversion/np.sqrt(20) # got 5e-5 by estimating by eye from plot_data_quick.py
data_ratio_err = data_error
#data_ratio_err = np.multiply(np.sqrt(np.power(np.divide(data_error,data_f),2) + np.power(np.divide(data_error,data_2f),2)), data_ratio)
#fit_ratio_err = np.multiply(np.sqrt(np.power(np.divide(err_f,fit_f),2) + np.power(np.divide(err_2f,fit_2f),2)), fit_ratio)

""" this gives the ratios as a function of the electric field
    so now let's extrapolate that data to get the desired ratio

    here Efield = m ratio + b 
    where m and b are in V/m """

fit_func = lambda x, m: m*x
m, cov = curve_fit(fit_func, data_ratio, Efield)
data_m = m[0]
print m[0]
data_b = 0.
data_err = 0.
#data_m, data_b, data_r, data_p, data_err = linregress(data_ratio, Efield)
#fit_m, fit_b, fit_r, fit_p, fit_err = linregress(fit_ratio, Efield)

fit_ratio = (np.array(Efield) - data_b)/data_m

""" Here's where it gets tricky, because max_limit is a function of the electric field: """
desired_measurement = 1e-21 # fractions of an electron charge
expected_measured_charge = 1.698e-4 * desired_measurement # C

# max_limit = expected_measured_charge * req_E # N
# desired_ratio = background / max_limit # unitless
# req_E = m * desired_ratio + b
#       = m * background / max_limit + b
#       = m * background / (expected_measured_charge * req_E) + b
#       = m * background / (expected_measured_charge * req_E) + b # V/m

# req_E**2 = m*background/expected_measured_charge + b*req_E
# 0 = req_E**2 - b*req_E - m*background/expected_measured_charge

# quadratic formula: x = (-b +/- sqrt(b^2 - 4*a*c))/(2*a)
four_a_c_data = 4*data_m*background/expected_measured_charge
#four_a_c_fit = 4*fit_m*background/expected_measured_charge
two_a = 2
b_data = data_b
#b_fit = fit_b
# req_E = (-b_quad +/- np.sqrt(b_quad**2 + four_a_c))/two_a
# but the negative solutions don't actually make sense so we'll ignore them.

req_E_data = (-b_data + np.sqrt(b_data**2 + four_a_c_data))/two_a
#req_E_fit = (-b_fit + np.sqrt(b_fit**2 + four_a_c_fit))/two_a # V/m

max_limit_data = expected_measured_charge * req_E_data # N
desired_ratio_data = background / max_limit_data # unitless
#max_limit_fit = expected_measured_charge * req_E_fit # N
#desired_ratio_fit = background / max_limit_fit # unitless

print ' '
print 'required electric field: ' + str(req_E_data) + ' +/- ' + str(data_err) + ' V/m'
#print 'required electric field from fit:  ' + str(req_E_fit) + ' +/- ' + str(fit_err) + ' V/m'
print ' ===== OR ===== '
print 'required electric field: ' + str(req_E_data/1e6) + ' +/- ' + str(data_err/1e6) + ' kV/mm'
#print 'required electric field from fit:  ' + str(req_E_fit/1e6) + ' +/- ' + str(fit_err/1e6) + ' kV/mm'
print ' '
print 'limit of amplitude at f from data: ' + str(max_limit_data) + ' N'
print 'ratio of amplitude at f to amplitude at 2f from data: ' + str(desired_ratio_data)
#print 'limit of amplitude at f from fit: ' + str(max_limit_fit) + ' N'
#print 'ratio of amplitude at f to amplitude at 2f from fit: ' + str(desired_ratio_fit)
print ' '

# Make the plots requested here

plot_amplitude_data_raw(path, Efield, data_f, data_2f, fit_f, fit_2f, err_f, err_2f) # plot the line and parabola

plot_highest_point(folder, data_error) # charge 43

plot_highest_point_z(folder2, data_error) # charge 45

plt.figure()
#plt.errorbar(np.array(Efield)/1e6, data_ratio, yerr = data_ratio_err, label = "data", fmt = 'b.')
plt.plot(np.array(Efield)/1e6, data_ratio, 'bo', label = "data")
plt.plot(np.array(Efield)/1e6, fit_ratio, 'r', label = "fit")
plt.legend(loc = 4)
plt.xlabel('Electric field [kV/mm]')
plt.xlim(-0.005, 0.355)
plt.ylabel('Ratio of force at 2f to force at f')
plt.title('Ratio of induced and permanent dipole')
plt.show()
